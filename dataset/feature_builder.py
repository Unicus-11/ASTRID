"""
feature_builder.py
IMPORTANT ========> VERY IMPORTANT : # NOTE: ( NEW/CURRENT/RECENT CHANGE)
# Cheng traffic-flow / queue features are retained as a supporting traffic-context
# module for the SIH system. They are NOT the primary features for the future
# trajectory prediction model.
#
# Keep this feature-engineering pipeline independent from the vehicle-level
# trajectory prediction features that will be built later. Do not remove or
# overwrite these Cheng features; they may be used later for congestion/queue
# context, model inputs, validation, or SIH visualization/explanation.


====================
ASTRID Prototype -- Feature Builder, LAYER 1 + LAYER 2

LAYER 1: camera-only state estimation.
    Directly observed camera fields + history-derived (past-only) deltas
    computed from those same fields.

LAYER 2: adds GPS/probe observations + signal-phase features + physics-
    derived features + Cheng-style per-probe critical-point features on
    top of everything Layer 1 already builds.
.....Impt -->  One thing worth double-checking later: q_u_cheng_veh_per_hour can spike to unrealistic values (thousands+) when T_CP2 - T_r is very small (a probe joins the queue almost immediately after red starts) — Eq.(12)'s denominator gets tiny. Your current sample (262, 12.4) looks fine, but scan the full column for extreme outliers before trusting it for training.
--------------------------------------------------------------------------
v0.7 -- TYPE I CP SELECTION CORRECTED AGAINST THE PAPER'S OWN ALGORITHM
--------------------------------------------------------------------------
This revision fixes a real bug in the v0.6 Type I CP selection, found by
re-checking the implementation line-by-line against Cheng et al.'s own
"Critical Points Filter for Various Purposes" steps (a)-(c):

  - The v0.6 docstring CLAIMED the code "walks backward from the first
    [stopped] CP to find the Type I CP". The v0.6 CODE actually did
    something different from both that claim and the paper: it scanned
    FORWARD from the start of the trajectory and took the *first* CP
    whose speed dropped below its immediate predecessor's speed, with NO
    further check. That is only half of the paper's step (b).

  - The paper's actual step (b) is a forward scan **with a validation +
    retry loop**: starting from the beginning of the trajectory, find
    the first CP i (chronologically) whose speed is below its immediate
    predecessor's speed. Then check whether v_i is >= every CP's speed
    from i up to the first stopped CP j1 (i.e., i's speed is not
    exceeded again before the vehicle actually stops -- confirming i is
    really where the deceleration regime begins, not just a local dip
    that recovers). If that check passes, i is the Type I CP. If it
    fails, DISCARD every CP from the start of the trajectory through i
    and repeat the search from the next remaining CP onward.

  - v0.6 implemented neither the validation check nor the discard/retry
    loop, and its docstring's "walk backward" description did not match
    what the code did either way. _select_type_i_ii_iii() below now
    implements the paper's forward-scan-with-validation-and-retry
    exactly as specified, and the docstring is corrected to describe the
    actual (paper-matching) algorithm rather than an inaccurate summary
    of it.

  - Everything else about the Cheng section (CP extraction Eq.(1)-(3),
    Type II/III selection, Eq.(9)/(11)/(12), causal/past-only
    availability on the feature grid) is functionally unchanged from
    v0.6 and re-verified against the paper while making this fix.

  - Added a small number of diagnostic Cheng features
    (cheng_t_cp2_s, cheng_l_cp2_m, cheng_t_cp3_s) alongside the three
    quantitative Cheng features, for debugging/QA -- e.g. checking
    whether a scenario/edge is producing Type II/III CPs at all, and
    when. These carry the same causal availability rules as the
    quantitative features they're diagnostics for (never available
    before the CP that produced them was actually observed).

--------------------------------------------------------------------------
v0.6 -- CHENG CRITICAL-POINT FEATURES FROM ACTUAL PER-PROBE TRAJECTORIES
--------------------------------------------------------------------------
gps_simulator.py (v0.6) now writes gps_p{TAG}_probe_trajectories.csv: a
1-second-resolution, per-probe, per-edge trajectory (approach ->
intersection -> outgoing edge) for every selected probe, with a stable
probe_id, distance_to_stopline_m populated only on approach edges, and
no fabricated timestamps. This is exactly the input Cheng, Qin, Jin &
Ran's method (TRB 2010 -- read in full; see prior revision's citation
audit, retained below) requires and which was previously unavailable
from the 5-second population aggregate alone.

This revision adds a dedicated Cheng/trajectory section
(load_probe_trajectories, extract_critical_points,
compute_cheng_queue_features, aggregate_cheng_features_to_grid) that:

  - Implements CP extraction per (probe_id, approach_edge) directly
    against Cheng et al.'s Eq.(1)-(3): a point is part of a uniform-
    motion or uniformly-accelerated regime (Eq 1/2) unless its speed is
    below the stopping threshold c_v,stop (Eq 3), which overrides and is
    treated as a distinct stopped segment. Thresholds follow the paper's
    own Numeric Experiment values: c_v = 3 mph, c_a = 3 ft/s^2,
    c_v,stop = 3 mph (converted to SI: c_v ~= 1.34 m/s, c_a ~= 0.914
    m/s^2, c_v,stop ~= 1.34 m/s).
  - Implements the paper's own Type I/II/III selection algorithm
    (Methodology section, "Critical Points Filter for Various Purposes",
    steps (a)-(c)): order CPs chronologically, find the run of CPs at or
    below c_v,stop; then FORWARD-SCAN from the start of the trajectory
    for the first CP whose speed drops below its immediate predecessor's,
    validate it against every CP up to the first stopped CP, discarding
    and retrying if the validation fails, to find the Type I CP (see the
    v0.7 note above -- this was corrected from v0.6); the last stopped CP
    is Type II and the following CP is Type III.
  - Implements Eq.(9)/(10) (local density from a single probe's Type I/
    Type II CP positions, and the re-derived upstream arrival flow used
    only as an intermediate quantity) and Eq.(12) (arrival flow q_u from
    a Type II CP's own position/timestamp) directly from L_CP1, L_CP2,
    T_CP1, T_CP2 read off that specific probe's own trajectory -- not
    from any population aggregate.
  - Implements Eq.(11), the maximum-queue-length-in-a-cycle estimate,
    per (approach_edge, red-phase cycle), using k_jam (already computed
    below for the physics section) as k_j and, where a Webster saturation
    flow reference is configured, q_s.
  - Does NOT implement Eq.(4)-(8) (shockwave/queue-front and queue-
    discharge speed derived from a two-point flux chord) as a Cheng
    feature: those require q_u/k_u or q_s/k_m/k_j on both sides of a
    (0,k_j) chord, several of which (v_dis, k_m specifically) are not
    reliably available here; ASTRID's own empirical
    estimated_queue_front_propagation_m_per_s already covers that role
    (see the physics section below) and is not replaced.
  - Does NOT implement Eq.(13)-(17) (initial-queue detection / piecewise
    multi-CP queue-growth weighting across n-1 Type II CPs within one
    cycle): this revision uses at most one Type II CP per probe per
    cycle (the standard case Eq. 11/12 cover); the n-CP piecewise
    extension is a genuine unimplemented feature, not something faked
    as equivalent.

  IMPORTANT CORRECTNESS NOTE ON EQ.(12): the paper's own Eq.(12),
        q_u = k_j * L_CP2 / (T_CP2 - T_r)
  requires T_r, the actual start-of-red time. ASTRID has this directly
  (red_duration_s / the signal controller's own phase state), unlike
  Cheng et al., who only had it via their own Eq.(7) red-time-detection
  step (itself dependent on a Type I CP). Feeding a *real* T_r into
  Eq.(12) is a strictly more direct, and more accurate, application of
  their equation, not a deviation from it.

  Earlier revisions of this file explicitly stated Cheng's method could
  not be implemented because gps_simulator.py exposed only a 5-second
  population aggregate with no continuous per-vehicle trajectory. That
  constraint no longer holds as of gps_simulator.py v0.6 and this
  revision; those comments/removed-feature entries are updated below
  rather than left stale.

Everything from the v0.5 literature-verification pass (Richards 1956,
Rempe et al. 2017, Kumari et al. 2025 -- none accessible in full text,
no equation used from any of them) is unchanged and retained below.

--------------------------------------------------------------------------
v0.5 -- LITERATURE VERIFICATION PASS (prior revision, retained)
--------------------------------------------------------------------------
This revision was preceded by an actual search-and-read pass over every
paper named in ASTRID's physics justification, not just a re-read of what
earlier code comments claimed those papers said. Result, paper by paper:

  Lighthill & Whitham (1955), "On kinematic waves II" -- FULL TEXT
  obtained and read directly (Proc. Roy. Soc. A 229, pp.317-345). Every LW
  citation below is checked against that text directly.

  Richards (1956), "Shock Waves on the Highway", Operations Research 4(1),
  pp.42-51 -- NOT accessible in full text. It is paywalled at the
  publisher (INFORMS/pubsonline.informs.org) and at JSTOR; no legitimate
  free full-text copy was found. Only the published abstract and
  secondary-literature summaries were available: it independently derives
  a fluid-continuum traffic theory the same year as LW (apparently without
  awareness of LW's paper), using an empirical speed-density relation, a
  "graph-shearing" wave construction, and specifically analyzes a traffic
  signal, finding a "threshold effect" where disturbances are minor for
  light traffic but build suddenly past a critical density. Because the
  full text was not available, NO specific equation, page number, or
  numbered result is attributed to Richards anywhere in this file --
  only this general, secondary-sourced characterization, and only where
  it does not carry any specific formula.

  Cheng, Qin, Jin & Ran -- the 2012 Journal of Intelligent Transportation
  Systems article was not directly accessible, but its immediate
  predecessor by the SAME FOUR AUTHORS on the SAME METHOD --
  "An Exploratory Shockwave Approach for Signalized Intersection
  Performance Measurements Using Probe Trajectories" (TRB 2010 Annual
  Meeting CD-ROM; the 2012 JITS article is this method's journal
  publication) -- WAS obtained in full text and read directly, and is now
  the basis of the v0.6 Cheng/trajectory section above, now that
  gps_simulator.py v0.6 actually exposes per-probe trajectories.

  Rempe, Kessler & Bogenberger (2017), "Fusing Probe Speed and Flow Data
  for Robust Short-Term Congestion Front Forecasts", 5th IEEE MT-ITS,
  Naples, DOI 10.1109/MTITS.2017.8005695 -- confirmed to exist (title,
  venue, DOI, page range 31-36 verified via multiple citing works and the
  authors' own publication lists), but the paper itself is paywalled at
  IEEE Xplore and no accessible full text or abstract was found. NO
  equation or specific claim is attributed to this paper anywhere in this
  file.

  "Lalitha et al. (2025, ICSCN)" -- correctly Kumari et al. (R. Lalitha is
  the last-listed author, not the first). NOT ACCESSIBLE IN FULL TEXT OR
  ABSTRACT. No concept, equation, or claim is attributed to it anywhere
  in this file.

Structural leakage guarantee: ground_truth/ is read in exactly ONE
function (build_labels), which writes to a separate labels file.
raw_output/vehicle_trajectories.csv is not read anywhere in this file.
gps_p{TAG}_probe_trajectories.csv (the per-PROBE observation, not ground
truth) is read only inside the Cheng/trajectory section, and only
gps-observable fields (edge, lane, distance-to-stopline while on an
approach, speed, acceleration) are used from it -- never a population-
level queue/density/flow computed elsewhere.

Reads (per scenario):
    scenario.json
    observations/camera_timeseries.csv           (Layer 1 + base of Layer 2)
    observations/assembled_observations_p{tag}.csv (Layer 2 -- from
                                                     observation_assembler.py)
    observations/gps_p{tag}_probe_trajectories.csv (Layer 2 -- Cheng/
                                                     trajectory section only)
    raw_output/tls_state.csv                      (Layer 2 signal features,
                                                     OPTIONAL -- signal
                                                     STATE, not vehicle data)
    ground_truth/state_timeseries.csv             (ONLY inside build_labels)

Writes:
    features/features_{layer}.csv
    features/labels_{layer}.csv
    features/feature_manifest_{layer}.json

Run:
    python dataset/feature_builder.py --scenario scenario_normal_balanced --layer layer1
    python dataset/feature_builder.py --scenario scenario_normal_balanced --layer layer2 --penetration 0.11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from trajectory_utils import SAMPLING_INTERVAL_S

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

DELTA_WINDOW_S = 30

# Project assumption -- NOT verified against this scenario's actual vType /
# car-following configuration, and NOT sourced from any supplied paper's
# equations. See compute_k_jam() docstring.
DEFAULT_EFFECTIVE_GAP_M = 2.5

DEFAULT_PENETRATION_RATE = 0.11  # ASTRID's primary GPS experiment

# Phase -> approach-group mapping, per the cross-referenced (not independently
# verified against raw <connection> data) resolution from the normal_controller.py
# audit: phases {0,2}=EW (edges 1i,2i), phases {4,6}=NS (edges 3i,4i).
PHASE_GREEN_GROUP = {0: "EW", 1: "EW", 2: "EW", 3: "EW", 4: "NS", 5: "NS", 6: "NS", 7: "NS"}
PHASE_IS_GREEN = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False, 6: True, 7: False}
GROUP_EDGES = {"EW": ["1i", "3i"], "NS": ["2i", "4i"]}
EDGE_GROUP = {e: g for g, edges in GROUP_EDGES.items() for e in edges}

# Fields no sensor-derived feature table may ever contain -- regression
# guard against ground truth (or a raw-trajectory reconstruction of it)
# leaking into features_*.csv. These belong only in labels_*.csv.
FORBIDDEN_GROUND_TRUTH_COLUMNS = {
    "queue_length_m", "queue_count", "queue_beyond_camera",
    "density_veh_per_km", "flow_veh_per_hour", "vehicle_count",
    "mean_speed_mps", "true_queue_length_m", "true_queue_beyond_camera",
}

CAMERA_REQUIRED_COLUMNS = [
    "timestamp", "approach_edge", "camera_range_m",
    "visible_vehicle_count", "visible_mean_speed_mps",
    "visible_queue_count", "visible_queue_length_m",
    "queue_reaches_camera_edge",
]
GPS_REQUIRED_COLUMNS = [
    "probe_count", "probe_mean_speed_mps",
    "probe_min_distance_to_stopline_m", "probe_max_distance_to_stopline_m",
]

# gps_p{TAG}_probe_trajectories.csv schema, as written by gps_simulator.py
# v0.6. Required for the Cheng/trajectory section only.
PROBE_TRAJECTORY_REQUIRED_COLUMNS = [
    "timestamp", "probe_id", "edge_id", "approach_edge", "lane_id",
    "speed_mps", "acceleration_mps2", "distance_to_stopline_m",
]

# ----------------------------------------------------------------------
# Cheng et al. (TRB 2010) CP-extraction thresholds -- their own Numeric
# Experiment values (c_v = 3 mph, c_a = 3 fpss, c_v,stop = 3 mph),
# converted to SI since ASTRID's trajectories are in m/s and m/s^2.
# 1 mph = 0.44704 m/s; 1 fpss = 0.3048 m/s^2.
# ----------------------------------------------------------------------
CHENG_C_V_MPS = 3.0 * 0.44704          # ~1.341 m/s -- uniform-motion speed-match threshold
CHENG_C_A_MPS2 = 3.0 * 0.3048          # ~0.914 m/s^2 -- uniform-accel match threshold
CHENG_C_V_STOP_MPS = 3.0 * 0.44704     # ~1.341 m/s -- "stopped" threshold (paper reuses c_v)

# ----------------------------------------------------------------------
# Eq.(12) plausibility guard -- NOT part of Cheng et al.'s equations.
# Prevents Eq.(12) from being applied to residual queues carried over
# from a previous under-cleared cycle.
# ----------------------------------------------------------------------
CHENG_MAX_PLAUSIBLE_APPROACH_SPEED_MPS = 20.0  # ~72 km/h


def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_no_ground_truth_leak(df: pd.DataFrame, where: str) -> None:
    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(df.columns)
    if leaked:
        raise ValueError(f"{where}: forbidden ground-truth-shaped column(s) present: {leaked}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series, min_denominator: float = 1e-9) -> pd.Series:
    """Elementwise division that returns NA (never inf/-inf/silently-wrong)
    wherever the denominator is not STRICTLY POSITIVE.

    v0.4 fix: previously masked on abs(denominator) > min_denominator,
    which accepted negative denominators. Every physical use of this
    helper in this file (camera_range_m, elapsed time in hours, a
    density/saturation-flow gap that is only physically meaningful when
    positive) requires denominator > 0, not merely denominator != 0 -- a
    negative denominator here would indicate an invalid/non-physical
    state (e.g. k_jam - k_A <= 0) and must produce NA, not a sign-flipped
    or otherwise misleading result."""
    denom = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    safe = denom.notna() & (denom > min_denominator)
    out = pd.Series(pd.NA, index=num.index, dtype="Float64")
    out[safe] = (num[safe] / denom[safe]).astype("Float64")
    return out


# ============================================================================
# LAYER 1 -- camera-only, directly observed + history-derived
# ============================================================================

def load_camera_observations(scenario_dir: Path) -> pd.DataFrame:
    path = scenario_dir / "observations" / "camera_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run sensors/camera_simulator.py first.")
    df = pd.read_csv(path)
    missing = [c for c in CAMERA_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"camera_timeseries.csv is missing required column(s): {missing}")
    _validate_no_ground_truth_leak(df, "camera_timeseries.csv")
    return df


def add_change_features(df: pd.DataFrame, columns: List[str], window_s: int = DELTA_WINDOW_S) -> pd.DataFrame:
    """Past-only history deltas over the given columns, per approach_edge.
    Works identically for camera columns (Layer 1) and GPS/probe columns
    (Layer 2) -- both are already on the same SAMPLING_INTERVAL_S grid in
    their respective source files, so this needs no resampling."""
    df = df.sort_values(["approach_edge", "timestamp"]).copy()
    delta_steps = max(1, window_s // SAMPLING_INTERVAL_S)
    g = df.groupby("approach_edge")
    for col in columns:
        if col in df.columns:
            df[f"{col}_change_{window_s}s"] = g[col].diff(delta_steps)
    return df


def add_occupancy_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """visible_queue_length_m / camera_range_m, clipped to [0, 1].

    This is a Layer 1 quantity: how much of the CAMERA'S OWN visible
    region the observed queue currently fills. Deliberately never
    interpreted as "fraction of the true queue" -- see the
    queue_reaches_camera_edge censoring note in build_feature_manifest()."""
    df = df.copy()
    fraction = _safe_divide(df["visible_queue_length_m"], df["camera_range_m"])
    df["visible_occupancy_fraction"] = fraction.clip(lower=0.0, upper=1.0)
    return df


def build_layer1_features(camera_df: pd.DataFrame) -> pd.DataFrame:
    features = add_change_features(camera_df, ["visible_queue_length_m", "visible_mean_speed_mps"])
    features = add_occupancy_fraction(features)
    return features


# ============================================================================
# LAYER 2 -- GPS/probe + signal + physics-derived, built on Layer 1
# ============================================================================

def load_assembled_observations(scenario_dir: Path, tag: str) -> pd.DataFrame:
    """Reads observation_assembler.py's output -- camera and GPS
    observations already merged and validated (grid-aligned, no duplicate
    keys, no ground-truth leakage) at the (timestamp, approach_edge) level."""
    path = scenario_dir / "observations" / f"assembled_observations_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} -- run sensors/observation_assembler.py "
            f"--scenario {scenario_dir.name} --penetration ... first."
        )
    df = pd.read_csv(path)
    missing = [c for c in CAMERA_REQUIRED_COLUMNS + GPS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"assembled_observations_{tag}.csv is missing required column(s): {missing}")
    _validate_no_ground_truth_leak(df, f"assembled_observations_{tag}.csv")
    return df


def load_tls_state(scenario_dir: Path) -> Optional[pd.DataFrame]:
    """Signal STATE (phase index, red/green string) -- not vehicle
    trajectory data. Allowed as a feature source (a real signal
    controller's own phase output is observable to a real deployment)."""
    path = scenario_dir / "raw_output" / "tls_state.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_signal_features(df: pd.DataFrame, tls_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()
    df["current_phase"] = pd.NA
    df["phase_elapsed_s"] = pd.NA
    df["is_green_for_approach"] = pd.NA
    df["red_duration_s"] = pd.NA

    if tls_df is None:
        print("  NOTE: no raw_output/tls_state.csv -- signal columns left NA. "
              "Re-run sumo/run_scenarios.py (with phase capture) for this scenario.")
        return df

    tls_df = tls_df.copy()
    tls_df["timestamp"] = tls_df["timestamp"].astype(float)
    tls_df = tls_df.sort_values("timestamp")
    tls_df["phase_changed"] = tls_df["phase"] != tls_df["phase"].shift(1)
    tls_df["phase_start_time"] = tls_df["timestamp"].where(tls_df["phase_changed"]).ffill()
    tls_df["phase_elapsed_s"] = tls_df["timestamp"] - tls_df["phase_start_time"]

    lookup = tls_df.set_index("timestamp")[["phase", "phase_elapsed_s"]]
    ts = df["timestamp"].astype(float)
    df["current_phase"] = ts.map(lookup["phase"])
    df["phase_elapsed_s"] = ts.map(lookup["phase_elapsed_s"])

    df["_group"] = df["approach_edge"].map(EDGE_GROUP)
    df["is_green_for_approach"] = df.apply(
        lambda r: (PHASE_GREEN_GROUP.get(r["current_phase"]) == r["_group"]
                   and PHASE_IS_GREEN.get(r["current_phase"], False))
        if pd.notna(r["current_phase"]) else pd.NA,
        axis=1,
    )
    df = df.drop(columns=["_group"])

    # red_duration_s: a proper consecutive-red-streak count per approach_edge.
    #
    # BUG FIX (was: group_break = (~is_red).groupby(...).cumsum()) --
    # cumsum() on (~is_red) increments AT every green row (a green row
    # contributes +1 to its own cumulative total), so a green row's
    # group_break value was identical to the red streak immediately
    # following it. That merged the trailing green row into the next red
    # group, making streak_start = that green row's timestamp instead of
    # the true first red row's timestamp -- red_duration_s was therefore
    # off by one grid step (SAMPLING_INTERVAL_S) for every single red
    # streak, and never actually reached 0 at the true first red row.
    # Concretely this broke build_signal_red_starts()'s `red_dur == 0`
    # check (it never matched anything -> empty red_starts -> Cheng
    # section never had a T_r to work from -> all Cheng columns NA), and
    # silently shifted estimated_hidden_queue_extension_m by ~5s on every
    # row, without raising any error.
    #
    # FIX: shift the (~is_red) series by one row before cumsum(), so a
    # green row's own "+1" lands on the NEXT row instead of itself. That
    # keeps the green row in its own (irrelevant) group and makes the
    # following red streak's group_break start cleanly at the true first
    # red row, so streak_start is the correct one and red_duration_s is
    # exactly 0 there.
    df = df.sort_values(["approach_edge", "timestamp"]).reset_index(drop=True)
    is_red = (df["is_green_for_approach"] == False)  # noqa: E712 -- NA-safe: NA==False -> False
    group_break = (~is_red).shift(1, fill_value=False).groupby(df["approach_edge"]).cumsum()
    streak_start = df.groupby(["approach_edge", group_break])["timestamp"].transform("min")
    df.loc[is_red, "red_duration_s"] = df.loc[is_red, "timestamp"] - streak_start[is_red]

    return df


def compute_k_jam(scenario: dict, vehicle_types_cfg: dict, effective_gap_m: float) -> float:
    """Jam density k_jam [veh/km].

    LITERATURE (Lighthill & Whitham 1955, p.322): jam headway (1/k_j) was
    empirically observed to be "only just greater than the average
    vehicle length". That is the extent of what the paper itself
    establishes -- a qualitative relationship between jam headway and
    vehicle length, with no specific formula given.

    ASTRID MODELING ASSUMPTION (this project, not LW's paper): built in
    the spirit of that finding, but the specific formula below is
    ASTRID's own choice:

        effective jam spacing = weighted average vehicle length
                                 (from scenario.json's vehicle_composition)
                                 + effective_gap_m (a project assumption,
                                 see DEFAULT_EFFECTIVE_GAP_M -- NOT verified
                                 against this scenario's actual vType /
                                 car-following configuration, and not
                                 sourced from any supplied paper)

        k_jam = 1000 / effective jam spacing

    vehicle_composition is the SCENARIO'S CONFIGURED design mix -- fixed
    for the whole scenario, exactly like camera_range_m or
    approach_length_m. It is scenario-level EXPERIMENTAL METADATA, not a
    per-timestamp sensor reading: it never varies row-to-row and carries
    no information about what happened at any particular timestamp, so it
    is not the kind of per-timestep "vehicle composition" the
    architecture rules prohibit.
    """
    composition = scenario["vehicle_composition"]
    avg_length_m = sum(composition[vt] * vehicle_types_cfg[vt]["length"] for vt in composition)
    effective_jam_spacing_m = avg_length_m + effective_gap_m
    if effective_jam_spacing_m <= 0:
        raise ValueError(
            f"Invalid k_jam inputs: avg_length_m={avg_length_m}, effective_gap_m={effective_gap_m}"
        )
    return round(1000.0 / effective_jam_spacing_m, 2)


def add_physics_features(
    df: pd.DataFrame, k_jam: float, saturation_flow_veh_per_hour: Optional[float] = None
) -> pd.DataFrame:
    """Physics-derived features built ONLY from sensor-observable columns
    already present on df (camera + GPS + signal). See module docstring
    for full citation grounding, including exactly which paper each piece
    is/is not sourced from.

    A. estimated_density_k_veh_per_km
        LW eq (2) concept, adapted to an instantaneous count/length
        snapshot OVER THE CAMERA'S OWN VISIBLE REGION
        (visible_vehicle_count / camera_range_m). An OBSERVED-REGION
        density; the true density of the full link (which may extend
        well past 150m) is not measured by this quantity.

    B. observed_flow_veh_per_hour
        q = k * v (LW eq (3), the space-mean-speed identity q=k*v),
        using this function's own density estimate and the camera's
        visible_mean_speed_mps. An AGGREGATE, MIXED-REGIME quantity over
        the camera's whole visible region -- kept as a general
        traffic-state feature, but NOT fed into a shock-speed formula
        (see C) because it does not represent a clean upstream state.

    C. estimated_queue_front_propagation_m_per_s
        The shock/queue-front speed, estimated EMPIRICALLY: the rate of
        change of visible_queue_length_m over DELTA_WINDOW_S. Positive =
        queue growing (front moving further from the stop line, i.e.
        propagating upstream). Valid only while queue_reaches_camera_edge
        is False for the current row. NOT sourced from any paper's
        equation -- a direct kinematic measurement of an observed
        boundary, standing in for the LW section 6 shock-speed concept
        without needing LW's clean-upstream-state assumption.

    D. estimated_hidden_queue_extension_m
        FIRST-ORDER, HELD-RATE APPROXIMATION -- not LW's exact eq (17)
        solution. The LAST OBSERVED, pre-censoring propagation rate (C)
        is forward-filled per approach_edge and held constant, then
        multiplied by red_duration_s once censoring begins. Assumes the
        queue does not shrink while still red and still censored.

    (Cheng et al.'s own per-probe critical-point method is implemented
    separately, in the Cheng/trajectory section below, now that
    gps_p{TAG}_probe_trajectories.csv provides actual per-probe
    trajectories. It is not duplicated here.)
    """
    df = df.copy()

    camera_range_km = pd.to_numeric(df["camera_range_m"], errors="coerce") / 1000.0
    df["estimated_density_k_veh_per_km"] = _safe_divide(df["visible_vehicle_count"], camera_range_km)

    speed_kmh = pd.to_numeric(df["visible_mean_speed_mps"], errors="coerce") * 3.6
    df["observed_flow_veh_per_hour"] = (
        df["estimated_density_k_veh_per_km"].astype("Float64") * speed_kmh.astype("Float64")
    )

    # -- C: empirical queue-front propagation rate --
    df = df.sort_values(["approach_edge", "timestamp"]).reset_index(drop=True)
    change_col = f"visible_queue_length_m_change_{DELTA_WINDOW_S}s"
    propagation_rate = pd.Series(pd.NA, index=df.index, dtype="Float64")
    if change_col in df.columns and "queue_reaches_camera_edge" in df.columns:
        raw_rate = pd.to_numeric(df[change_col], errors="coerce") / DELTA_WINDOW_S
        currently_uncensored = df["queue_reaches_camera_edge"] == False  # noqa: E712
        propagation_rate[currently_uncensored] = raw_rate[currently_uncensored].astype("Float64")
    df["estimated_queue_front_propagation_m_per_s"] = propagation_rate

    # Last known (pre-censoring) rate, forward-filled per approach_edge, for
    # use while the queue is currently censored -- see (D) above.
    last_known_rate = df.groupby("approach_edge")["estimated_queue_front_propagation_m_per_s"].ffill()

    # -- D: hidden-queue extension, held-rate extrapolation --
    df["estimated_hidden_queue_extension_m"] = pd.NA
    if "is_green_for_approach" in df.columns and "red_duration_s" in df.columns and "queue_reaches_camera_edge" in df.columns:
        is_censored = df["queue_reaches_camera_edge"] == True  # noqa: E712
        is_red = df["is_green_for_approach"] == False  # noqa: E712
        red_duration = pd.to_numeric(df["red_duration_s"], errors="coerce")
        has_red_duration = red_duration.notna() & (red_duration >= 0)
        rate_for_extrapolation = last_known_rate.clip(lower=0)
        has_rate = rate_for_extrapolation.notna()

        mask = is_censored & is_red & has_red_duration & has_rate
        if mask.any():
            df.loc[mask, "estimated_hidden_queue_extension_m"] = (
                rate_for_extrapolation[mask] * red_duration[mask]
            )

    return df


# ============================================================================
# CHENG / TRAJECTORY SECTION -- per-probe critical-point extraction
# ============================================================================
#
# This section implements Cheng, Qin, Jin & Ran's method directly against
# gps_p{TAG}_probe_trajectories.csv (gps_simulator.py v0.6): a real,
# per-probe, per-second, per-edge trajectory. It reads ONLY that GPS
# observation file -- never raw_output/vehicle_trajectories.csv, never
# ground_truth/.
#
# Equations implemented (numbering matches the TRB 2010 paper):
#   Eq.(1)-(3)  CP extraction (uniform motion / uniform acceleration /
#               stopped-segment override)
#   "Critical Points Filter" steps (a)-(c): Type I/II/III CP selection
#   Eq.(9)/(10) local density k_CP1 from a single probe's Type I/Type II
#               CP positions, feeding the re-derived arrival-flow form
#   Eq.(12)     q_u = k_j * L_CP2 / (T_CP2 - T_r), using ASTRID's real
#               signal-controller T_r rather than Cheng et al.'s own
#               detected T_r (see module docstring note)
#   Eq.(11)     L_q = q_s * q_u * (T_g - T_r) / (k_j * (q_s - q_u)),
#               applied per (approach_edge, cycle) using every probe's
#               Type II CP available in that cycle (max over probes,
#               since Eq. 11/12 as written are single-probe formulas and
#               ASTRID has no principled way to combine multiple probes'
#               q_u estimates into one without inventing a weighting
#               scheme the paper does not specify for this case)
#
# NOT implemented (see module docstring for why): Eq.(4)-(8) shockwave-
# speed-from-flux-chord; Eq.(13)-(17) initial-queue detection and the
# n-CP piecewise queue-growth weighting.
# ============================================================================

def load_probe_trajectories(scenario_dir: Path, tag: str) -> pd.DataFrame:
    """Reads gps_p{TAG}_probe_trajectories.csv -- the per-probe, per-
    second, any-edge GPS trajectory written by gps_simulator.py v0.6.
    This is a GPS OBSERVATION (not ground truth): it reports only what a
    real onboard GPS/IMU device could contribute for its own vehicle."""
    path = scenario_dir / "observations" / f"gps_{tag}_probe_trajectories.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} -- run sensors/gps_simulator.py "
            f"--scenario {scenario_dir.name} --penetration ... first (v0.6+ required "
            f"for gps_{{tag}}_probe_trajectories.csv)."
        )
    df = pd.read_csv(path)
    missing = [c for c in PROBE_TRAJECTORY_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"gps_{tag}_probe_trajectories.csv is missing required column(s): {missing}")
    _validate_no_ground_truth_leak(df, f"gps_{tag}_probe_trajectories.csv")
    return df


def _extract_cps_single_trajectory(traj: pd.DataFrame) -> pd.DataFrame:
    """Cheng et al. Eq.(1)-(3) CP extraction for ONE probe's trajectory
    (already sorted by timestamp, single probe_id).

    A run of consecutive points belongs to one regime -- uniform motion
    (Eq 1: |v_i - median(v)| < c_v) or uniform acceleration (Eq 2:
    |a_i - median(a)| < c_a) -- until a point breaks both conditions,
    which becomes a new CP (a regime boundary). Eq.(3) overrides both:
    any point with v_i < c_v,stop is treated as part of a distinct
    "stopped" segment regardless of what Eq 1/2 would say, so the exact
    moment of joining/leaving a queue is never absorbed into a coarser
    regime.

    This is a straightforward sequential-segmentation implementation of
    the paper's own description (max n such that Eq 1 or 2 holds for all
    points since the last CP; Eq 3 short-circuits and starts a stopped
    segment). Returns the subset of traj's rows that are CPs, with a
    boolean is_stopped_cp column marking Eq.(3) points.
    """
    traj = traj.reset_index(drop=True)
    n = len(traj)
    if n == 0:
        return traj.iloc[0:0].assign(is_stopped_cp=pd.Series(dtype=bool))

    speeds = traj["speed_mps"].to_numpy(dtype=float)
    accels = traj["acceleration_mps2"].to_numpy(dtype=float)

    cp_indices = [0]
    is_stopped_cp = [bool(speeds[0] < CHENG_C_V_STOP_MPS)]

    seg_start = 0
    in_stopped_segment = speeds[0] < CHENG_C_V_STOP_MPS

    for i in range(1, n):
        stopped_now = speeds[i] < CHENG_C_V_STOP_MPS

        if stopped_now and not in_stopped_segment:
            # Eq.(3) override: enter a new stopped segment -- this point
            # is a CP (the point where the vehicle drops below c_v,stop).
            cp_indices.append(i)
            is_stopped_cp.append(True)
            seg_start = i
            in_stopped_segment = True
            continue

        if in_stopped_segment and not stopped_now:
            # Leaving the stopped segment -- this point is a CP (the
            # first point back above c_v,stop).
            cp_indices.append(i)
            is_stopped_cp.append(False)
            seg_start = i
            in_stopped_segment = False
            continue

        if in_stopped_segment:
            # Still stopped -- no new CP; stopped points are all
            # implicitly part of the current stopped segment.
            continue

        # Not stopped: check Eq.(1)/(2) against the median of the
        # current open segment [seg_start, i].
        seg_v = speeds[seg_start:i + 1]
        seg_a = accels[seg_start:i + 1]
        med_v = float(np.median(seg_v))
        med_a = float(np.median(seg_a))

        uniform_motion_ok = abs(speeds[i] - med_v) < CHENG_C_V_MPS
        uniform_accel_ok = abs(accels[i] - med_a) < CHENG_C_A_MPS2

        if not (uniform_motion_ok or uniform_accel_ok):
            cp_indices.append(i)
            is_stopped_cp.append(False)
            seg_start = i

    out = traj.iloc[cp_indices].copy()
    out["is_stopped_cp"] = is_stopped_cp
    return out


def _select_type_i_ii_iii(cps: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
    """Cheng et al.'s "Critical Points Filter" selection, steps (a)-(c),
    applied to one probe's already-extracted, chronologically-ordered CPs
    for one signal cycle's worth of trajectory.

    v0.7 CORRECTION: earlier revisions of this function claimed (in their
    docstring) to "walk backward" from the first stopped CP, but the code
    actually did a plain forward scan with no validation step -- neither
    of which is what the paper's step (b) actually specifies. This is the
    paper's algorithm, implemented directly:

    (a) Order all CPs from this probe chronologically and find the CPs
        with speed < c_v,stop (the stopped-segment CPs, indices
        j1, j2, ... jm). If none exist, this probe was never stopped by a
        standing queue -- no Type II/III CP exists for it this cycle.
    (b) Let p = j1 (the first stopped CP). FORWARD-SCAN chronologically
        from the start of the trajectory for the first CP i (i.e. the
        first CP whose speed is less than its immediate predecessor CP's
        speed). Then validate: if CP i's speed is >= every CP's speed
        from i through j1 (i.e. no later CP before the stop exceeds i's
        speed again), i is the Type I CP -- go to (c). If the validation
        fails, DISCARD every CP from the start of the trajectory through
        i (they were a false start / local fluctuation, not the real
        beginning of the deceleration regime) and repeat this step,
        continuing the forward scan from the next remaining CP.
    (c) CP jm (the LAST stopped CP) is the Type II CP; the CP immediately
        after it is the Type III CP.

    Returns None if this trajectory has no Type II CP (never joined a
    queue this cycle) -- a valid, common outcome, not an error. Type I
    may still be unresolved (None) even when Type II/III are found, e.g.
    if the probe's observed trajectory begins already mid-deceleration
    (no earlier CP to serve as the search's starting point) or if no
    candidate ever passes the validation check -- both are genuine,
    expected "Cheng not fully resolvable for this probe/cycle" outcomes,
    not errors, and are left as NA downstream rather than fabricated.
    """
    cps = cps.reset_index(drop=True)
    stopped_mask = cps["speed_mps"] < CHENG_C_V_STOP_MPS
    if not stopped_mask.any():
        return None

    stopped_idx = cps.index[stopped_mask].tolist()
    first_stopped = stopped_idx[0]   # j1
    last_stopped = stopped_idx[-1]   # jm

    # (b): forward scan + validate + discard/retry, exactly as the paper
    # describes. search_from is the earliest index still "in play" as a
    # candidate predecessor -- everything before a failed candidate gets
    # discarded on retry.
    type_i_idx = None
    search_from = 1
    while search_from <= first_stopped:
        candidate_i = None
        for i in range(search_from, first_stopped + 1):
            if cps.loc[i, "speed_mps"] < cps.loc[i - 1, "speed_mps"]:
                candidate_i = i
                break
        if candidate_i is None:
            # No further CP decelerates relative to its immediate
            # predecessor before the stop -- Type I is not resolvable
            # for this probe/cycle (e.g. the probe's observed trajectory
            # starts already inside the deceleration).
            break
        window = cps.loc[candidate_i:first_stopped, "speed_mps"]
        if cps.loc[candidate_i, "speed_mps"] >= window.max():
            # Validated: candidate_i's speed is never exceeded again
            # before the stop, so it is genuinely where the deceleration
            # regime begins.
            type_i_idx = candidate_i
            break
        # Validation failed -- candidate_i was a local dip that recovers,
        # not the true start of deceleration. Discard everything through
        # candidate_i and keep scanning forward from the next CP.
        search_from = candidate_i + 1

    # (c): Type III is the CP immediately after the last stopped CP, if
    # one exists in this window (the probe's trajectory may end at/near
    # the stop, e.g. if the observation window cuts off before green).
    type_iii_idx = last_stopped + 1 if (last_stopped + 1) < len(cps) else None

    result = {
        "type_i": cps.loc[type_i_idx] if type_i_idx is not None else None,
        "type_ii": cps.loc[last_stopped],
        "type_iii": cps.loc[type_iii_idx] if type_iii_idx is not None else None,
    }
    return result


def _cycle_id_for_timestamp(ts: float, red_starts: List[float]) -> Optional[int]:
    """Assigns a cycle index to a timestamp: the index of the latest
    red_starts entry at or before ts. red_starts must be sorted
    ascending. Returns None if ts precedes the first known red start
    (no cycle boundary observed yet -- not fabricated)."""
    idx = None
    for i, r in enumerate(red_starts):
        if r <= ts:
            idx = i
        else:
            break
    return idx

def extract_critical_points(
    probe_traj_df: pd.DataFrame, signal_df: pd.DataFrame
) -> pd.DataFrame:
    """Runs CP extraction + Type I/II/III selection for every
    (probe_id, approach_edge, cycle) combination present in
    probe_traj_df, using ONLY that probe's own past-and-present
    trajectory rows.

    A probe's post-stop-line rows (internal/outgoing edges,
    approach_edge forward-filled) are intentionally included so Type III
    queue-discharge acceleration can be identified.

    Returns one row per resolved Type II CP with columns:
    probe_id, approach_edge, cycle_id,
    T_CP1, L_CP1, V_CP1,
    T_CP2, L_CP2,
    T_CP3, L_CP3,
    T_r.
    """
    required_cols = {"probe_id", "approach_edge", "timestamp"}
    if not required_cols.issubset(probe_traj_df.columns):
        raise ValueError(
            f"extract_critical_points: missing columns "
            f"{required_cols - set(probe_traj_df.columns)}"
        )

    rows = []

    for approach_edge, red_starts in signal_df.items():
        if not red_starts:
            continue

        edge_traj = probe_traj_df[
            probe_traj_df["approach_edge"] == approach_edge
        ]

        if edge_traj.empty:
            continue

        for probe_id, probe_df in edge_traj.groupby("probe_id"):
            probe_df = (
                probe_df
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            # Assign each row to a cycle using ONLY red_starts at or
            # before that row's own timestamp. A later cycle's red start
            # can therefore never attribute an earlier trajectory row
            # to a later cycle.
            probe_df["_cycle_id"] = probe_df["timestamp"].apply(
                lambda t: _cycle_id_for_timestamp(t, red_starts)
            )

            # Track this probe's speed at the end of the PREVIOUS cycle
            # chunk. If the probe was already stopped at that boundary,
            # the first stopped CP of the next chunk may represent the
            # continuation of the same physical stop rather than a new
            # arrival event.
            #
            # groupby() on the ascending integer cycle_id iterates in
            # chronological order, so this carries forward correctly.
            prev_cycle_last_speed = None

            for cycle_id, cycle_df in probe_df.groupby("_cycle_id"):
                if cycle_id is None or pd.isna(cycle_id):
                    continue

                cycle_df = (
                    cycle_df
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                # If there is only one trajectory row in this chunk,
                # there is not enough information for CP extraction.
                # Still carry its speed forward because it is the final
                # observed speed of the previous chunk.
                if len(cycle_df) < 2:
                    if len(cycle_df):
                        prev_cycle_last_speed = cycle_df["speed_mps"].iloc[-1]
                    continue

                # Extract the Cheng critical points from this probe's
                # trajectory only.
                cps = _extract_cps_single_trajectory(cycle_df)

                # ------------------------------------------------------
                # CONTINUITY FIX
                # ------------------------------------------------------
                # If this probe was already stopped at the end of the
                # previous cycle chunk, and the first CP of this chunk is
                # itself a stopped CP, that first CP is the continuation
                # of the existing stop.
                #
                # _extract_cps_single_trajectory() always emits its first
                # trajectory point as a CP. Without this guard, that
                # carried-over stopped vehicle can be mistaken for a new
                # Type II arrival immediately after T_r, producing an
                # unrealistically small:
                #
                #     T_CP2 - T_r
                #
                # and therefore an extreme q_u through Eq. (12).
                #
                # Only index 0 is removed. A genuinely new stopped segment
                # later in the same cycle remains untouched.
                carried_over_stopped = (
                    prev_cycle_last_speed is not None
                    and prev_cycle_last_speed < CHENG_C_V_STOP_MPS
                )

                if (
                    carried_over_stopped
                    and len(cps) > 0
                    and bool(cps.iloc[0]["is_stopped_cp"])
                ):
                    cps = cps.iloc[1:].reset_index(drop=True)

                # Update the carry-forward state AFTER evaluating the
                # continuity condition for this cycle.
                prev_cycle_last_speed = cycle_df["speed_mps"].iloc[-1]

                # Run Type I / II / III selection only after the
                # continuity correction.
                selected = _select_type_i_ii_iii(cps)

                if selected is None or selected["type_ii"] is None:
                    continue

                type_i = selected["type_i"]
                type_ii = selected["type_ii"]
                type_iii = selected["type_iii"]

                rows.append(
                    {
                        "probe_id": probe_id,
                        "approach_edge": approach_edge,
                        "cycle_id": int(cycle_id),
                        "T_r": red_starts[int(cycle_id)],

                        # Type I: beginning of deceleration.
                        "T_CP1": (
                            type_i["timestamp"]
                            if type_i is not None
                            else pd.NA
                        ),
                        "L_CP1": (
                            type_i["distance_to_stopline_m"]
                            if type_i is not None
                            else pd.NA
                        ),
                        "V_CP1": (
                            type_i["speed_mps"]
                            if type_i is not None
                            else pd.NA
                        ),

                        # Type II: stopped/queue-entry CP.
                        "T_CP2": type_ii["timestamp"],
                        "L_CP2": type_ii["distance_to_stopline_m"],

                        # Type III: discharge/acceleration CP.
                        "T_CP3": (
                            type_iii["timestamp"]
                            if type_iii is not None
                            else pd.NA
                        ),
                        "L_CP3": (
                            type_iii["distance_to_stopline_m"]
                            if type_iii is not None
                            else pd.NA
                        ),
                    }
                )

    return pd.DataFrame(rows)


def compute_cheng_queue_features(
    cp_df: pd.DataFrame,
    k_jam: float,
    saturation_flow_veh_per_hour: Optional[float],
) -> pd.DataFrame:
    """Applies Cheng et al.'s Eq.(9)/(10)/(12)/(11) to each resolved
    Type II CP row from extract_critical_points().

    Eq.(9): k_CP1 = k_j * (L_CP2 / L_CP1)

    Eq.(12): q_u = k_j * L_CP2 / (T_CP2 - T_r)

    Eq.(11): L_q = q_s * q_u * (T_g - T_r)
              / (k_j * (q_s - q_u))

    A plausibility guard masks q_u when the implied average approach
    speed L_CP2 / (T_CP2 - T_r) is physically impossible. This guard
    is an ASTRID sanity check, not part of Cheng et al.'s equations.
    """
    df = cp_df.copy()

    if df.empty:
        df["q_u_cheng_veh_per_hour"] = pd.Series(dtype="Float64")
        df["k_cp1_cheng_veh_per_km"] = pd.Series(dtype="Float64")
        df["max_queue_length_cheng_m"] = pd.Series(dtype="Float64")
        df["cheng_t_cp2_s"] = pd.Series(dtype="Float64")
        df["cheng_l_cp2_m"] = pd.Series(dtype="Float64")
        df["cheng_t_cp3_s"] = pd.Series(dtype="Float64")
        return df

    l_cp1 = pd.to_numeric(df["L_CP1"], errors="coerce")
    l_cp2 = pd.to_numeric(df["L_CP2"], errors="coerce")
    t_cp2 = pd.to_numeric(df["T_CP2"], errors="coerce")
    t_r = pd.to_numeric(df["T_r"], errors="coerce")

    df["cheng_t_cp2_s"] = t_cp2.astype("Float64")
    df["cheng_l_cp2_m"] = l_cp2.astype("Float64")
    df["cheng_t_cp3_s"] = (
        pd.to_numeric(df["T_CP3"], errors="coerce")
        .astype("Float64")
    )

    # Eq.(9)
    df["k_cp1_cheng_veh_per_km"] = (
        k_jam * _safe_divide(l_cp2, l_cp1)
    ).astype("Float64")

    # Eq.(12)
    elapsed_s = t_cp2 - t_r
    l_cp2_km = l_cp2 / 1000.0

    # Sanity-check only -- NOT part of Cheng et al.'s equations.
    # Guards against Eq.(12) being applied when its undersaturated-cycle
    # assumption is violated by a residual queue from the previous cycle.
    implied_speed_mps = _safe_divide(l_cp2, elapsed_s)

    plausible = (
        implied_speed_mps.notna()
        & (implied_speed_mps >= 0.0)
        & (
            implied_speed_mps
            <= CHENG_MAX_PLAUSIBLE_APPROACH_SPEED_MPS
        )
    )

    q_u_veh_per_s = k_jam * _safe_divide(l_cp2_km, elapsed_s)

    df["q_u_cheng_veh_per_hour"] = (
        q_u_veh_per_s.astype("Float64") * 3600.0
    )

    # Invalid / physically impossible Eq.(12) cases are not usable.
    df.loc[~plausible, "q_u_cheng_veh_per_hour"] = pd.NA

    # Eq.(11)
    if (
        saturation_flow_veh_per_hour is not None
        and saturation_flow_veh_per_hour > 0
    ):
        q_s = float(saturation_flow_veh_per_hour)
        q_u = df["q_u_cheng_veh_per_hour"]

        t_g = pd.to_numeric(df["T_CP3"], errors="coerce")

        # Convert seconds to hours because q_s and q_u are veh/hour.
        t_g_minus_t_r_h = (t_g - t_r) / 3600.0

        denom = k_jam * (q_s - q_u.astype("Float64"))

        numer = (
            q_s
            * q_u.astype("Float64")
            * t_g_minus_t_r_h.astype("Float64")
        )

        df["max_queue_length_cheng_m"] = (
        _safe_divide(numer, denom) * 1000.0
        ).astype("Float64")
        
        
    else:
        df["max_queue_length_cheng_m"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Float64",
        )

    return df
### This is updated 
def aggregate_cheng_features_to_grid(
    cheng_df: pd.DataFrame,
    grid_index: pd.DataFrame,
    red_starts: Dict[str, List[float]],
) -> pd.DataFrame:
    """Aggregates coherent Cheng events onto the 5-second feature grid.

    One Cheng event is selected per grid row. CP2-derived fields become
    available at T_CP2. CP3-derived fields become available later, at
    T_CP3, but always from the SAME Cheng event.

    Cycle assignment uses the original signal red_starts.
    """
    t_cp2_cols = [
        "q_u_cheng_veh_per_hour",
        "k_cp1_cheng_veh_per_km",
        "cheng_t_cp2_s",
        "cheng_l_cp2_m",
    ]

    t_cp3_cols = [
        "max_queue_length_cheng_m",
        "cheng_t_cp3_s",
    ]

    feature_cols = t_cp2_cols + t_cp3_cols

    out = (
        grid_index[["timestamp", "approach_edge"]]
        .drop_duplicates()
        .sort_values(["approach_edge", "timestamp"])
        .reset_index(drop=True)
    )

    for col in feature_cols:
        out[col] = pd.NA

    if cheng_df.empty:
        for col in feature_cols:
            out[col] = out[col].astype("Float64")
        return out

    cheng_df = cheng_df.copy()

    for col in ["T_CP2", "T_CP3", "T_r", "cycle_id"]:
        if col in cheng_df.columns:
            cheng_df[col] = pd.to_numeric(
                cheng_df[col],
                errors="coerce",
            )

    for approach_edge, edge_cheng in cheng_df.groupby("approach_edge"):
        edge_red_starts = red_starts.get(approach_edge, [])
        if not edge_red_starts:
            continue

        edge_mask = out["approach_edge"] == approach_edge
        edge_grid = out.loc[edge_mask].copy()

        events = edge_cheng.dropna(
            subset=["T_CP2", "cycle_id"]
        ).copy()

        if events.empty:
            continue

        def _values_for_row(row_ts: float) -> Dict[str, object]:
            row_cycle = _cycle_id_for_timestamp(
                row_ts,
                edge_red_starts,
            )

            if row_cycle is None:
                return {col: pd.NA for col in feature_cols}

            # ----------------------------------------------------------
            # Select ONE canonical Cheng event using T_CP2.
            # ----------------------------------------------------------
            candidates = events[
                (events["cycle_id"] == row_cycle)
                & (events["T_CP2"] <= row_ts)
            ]

            if candidates.empty:
                return {col: pd.NA for col in feature_cols}

            event = candidates.sort_values(
                "T_CP2"
            ).iloc[-1]

            result = {
                col: event[col]
                for col in t_cp2_cols
            }

            # ----------------------------------------------------------
            # CP3 fields must come from THIS SAME event.
            # ----------------------------------------------------------
            t_cp3 = event["T_CP3"]

            if pd.notna(t_cp3) and t_cp3 <= row_ts:
                for col in t_cp3_cols:
                    result[col] = event[col]
            else:
                for col in t_cp3_cols:
                    result[col] = pd.NA

            return result

        values = edge_grid["timestamp"].apply(_values_for_row)

        for col in feature_cols:
            edge_grid[col] = values.apply(lambda x, _col=col: x[_col])

        out.loc[
            edge_mask,
            feature_cols,
        ] = edge_grid[feature_cols].values

    for col in feature_cols:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        ).astype("Float64")

    return out

def build_signal_red_starts(signal_features_df: pd.DataFrame, approach_edges: List[str]) -> Dict[str, List[float]]:
    """Builds, per approach_edge, the sorted list of red-phase start
    timestamps -- reusing the is_green_for_approach / red_duration_s
    signal features add_signal_features() already computed on the grid,
    rather than re-deriving signal-phase logic in the Cheng section.

    A red start is identified as a grid row where is_green_for_approach
    is False and red_duration_s == 0 (the first row of a red streak, per
    add_signal_features()'s own consecutive-red-streak construction).
    Returns {} entries for edges with no signal data available (tls_state
    was missing) -- Cheng features are then simply left NA for that edge,
    same as the rest of the signal-dependent pipeline."""
    result: Dict[str, List[float]] = {e: [] for e in approach_edges}
    if "is_green_for_approach" not in signal_features_df.columns:
        return result

    for approach_edge, edge_df in signal_features_df.groupby("approach_edge"):
        edge_df = edge_df.sort_values("timestamp")
        is_red = edge_df["is_green_for_approach"] == False  # noqa: E712
        red_dur = pd.to_numeric(edge_df["red_duration_s"], errors="coerce")
        red_start_mask = is_red & (red_dur == 0)
        starts = sorted(edge_df.loc[red_start_mask, "timestamp"].astype(float).tolist())
        if approach_edge in result:
            result[approach_edge] = starts

    return result


def add_cheng_trajectory_features(
    df: pd.DataFrame,
    probe_traj_df: pd.DataFrame,
    approach_edges: List[str],
    k_jam: float,
    saturation_flow_veh_per_hour: Optional[float],
) -> pd.DataFrame:
    """Orchestrates the Cheng/trajectory section: builds red-phase start
    times from df's own already-computed signal features, extracts CPs
    and Type I/II/III selections from probe_traj_df, applies Cheng's
    Eq.(9)/(11)/(12), and aggregates the result back onto df's
    (timestamp, approach_edge) grid.

    If probe_traj_df is empty (e.g. no probes observed at all this
    scenario/penetration), all Cheng columns are added as all-NA rather
    than the section being skipped -- so the feature schema is identical
    across scenarios regardless of whether any probe happened to produce
    a resolvable CP.
    """
    df = df.copy()
    feature_cols = [
        "q_u_cheng_veh_per_hour", "k_cp1_cheng_veh_per_km", "max_queue_length_cheng_m",
        "cheng_t_cp2_s", "cheng_l_cp2_m", "cheng_t_cp3_s",
    ]

    red_starts = build_signal_red_starts(df, approach_edges)

    if probe_traj_df.empty or not any(red_starts.values()):
        for col in feature_cols:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="Float64")
        return df

    cp_df = extract_critical_points(probe_traj_df, red_starts)
    cheng_df = compute_cheng_queue_features(cp_df, k_jam, saturation_flow_veh_per_hour)

    aggregated = aggregate_cheng_features_to_grid(
    cheng_df,
    df[["timestamp", "approach_edge"]],
    red_starts, # Also updated this
)

    df = df.merge(aggregated, on=["timestamp", "approach_edge"], how="left", validate="one_to_one")
    
    return df


def build_layer2_features(
    assembled_df: pd.DataFrame,
    tls_df: Optional[pd.DataFrame],
    probe_traj_df: pd.DataFrame,
    approach_edges: List[str],
    k_jam: float,
    saturation_flow_veh_per_hour: Optional[float],
) -> pd.DataFrame:
    features = add_change_features(assembled_df, ["visible_queue_length_m", "visible_mean_speed_mps"])
    features = add_occupancy_fraction(features)
    features = add_change_features(
        features, ["probe_count", "probe_max_distance_to_stopline_m"], window_s=DELTA_WINDOW_S
    )
    features = add_signal_features(features, tls_df)
    features = add_physics_features(features, k_jam, saturation_flow_veh_per_hour)
    features = add_cheng_trajectory_features(
        features, probe_traj_df, approach_edges, k_jam, saturation_flow_veh_per_hour
    )
    return features


# ============================================================================
# Labels -- ground truth is read ONLY here, and only here (both layers)
# ============================================================================

def build_labels(scenario_dir: Path, horizon_s: Optional[int] = None) -> pd.DataFrame:
    gt_path = scenario_dir / "ground_truth" / "state_timeseries.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path} -- run dataset/ground_truth.py first.")
    gt = pd.read_csv(gt_path).sort_values(["approach_edge", "timestamp"]).copy()

    labels = gt[["timestamp", "approach_edge", "queue_length_m", "queue_beyond_camera"]].rename(
        columns={"queue_length_m": "true_queue_length_m", "queue_beyond_camera": "true_queue_beyond_camera"}
    )

    if horizon_s:
        horizon_steps = max(1, horizon_s // SAMPLING_INTERVAL_S)
        labels["true_queue_length_future_m"] = gt.groupby("approach_edge")["queue_length_m"].shift(-horizon_steps)
        labels["prediction_horizon_s"] = horizon_s

    return labels


# ============================================================================
# Feature manifest
# ============================================================================

def build_feature_manifest(layer: str, has_tls: bool) -> dict:
    manifest = {
        "layer": layer,
        "timestamp": {"kind": "index"},
        "approach_edge": {"kind": "index"},
        "camera_range_m": {"kind": "observed", "source": "camera",
            "note": "fixed network property, carried through for downstream use"},
        "visible_vehicle_count": {"kind": "observed", "source": "camera"},
        "visible_mean_speed_mps": {"kind": "observed", "source": "camera"},
        "visible_queue_count": {"kind": "observed", "source": "camera"},
        "visible_queue_length_m": {"kind": "observed", "source": "camera",
            "note": "furthest queued vehicle VISIBLE to the camera -- never the true total queue length"},
        "visible_occupancy_fraction": {"kind": "derived", "source": "camera",
            "note": "fraction of the camera's OWN 150m range the visible queue fills -- not fraction of true queue"},
        "queue_reaches_camera_edge": {"kind": "observed", "source": "camera",
            "note": "BOUNDARY/CENSORING indicator: True means the visible queue reaches ~the camera's edge, "
                    "which does NOT mean the true queue equals camera_range_m."},
        f"visible_queue_length_m_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "camera history (past only)"},
        f"visible_mean_speed_mps_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "camera history (past only)"},
    }

    if layer == "layer2":
        manifest.update({
            "probe_count": {"kind": "observed", "source": "gps"},
            "probe_mean_speed_mps": {"kind": "observed", "source": "gps"},
            "probe_min_distance_to_stopline_m": {"kind": "observed", "source": "gps"},
            "probe_max_distance_to_stopline_m": {"kind": "observed", "source": "gps",
                "note": "a real probe sighting; may exceed camera_range_m by design -- GPS is not range-limited"},
            f"probe_count_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "gps history (past only)"},
            f"probe_max_distance_to_stopline_m_change_{DELTA_WINDOW_S}s": {
                "kind": "derived", "source": "gps history (past only)"},
            "current_phase": {"kind": "observed" if has_tls else "unavailable", "source": "signal controller"},
            "phase_elapsed_s": {"kind": "observed" if has_tls else "unavailable", "source": "signal controller"},
            "is_green_for_approach": {"kind": "derived" if has_tls else "unavailable",
                "note": "inherits the not-independently-verified phase-to-approach mapping caveat"},
            "red_duration_s": {"kind": "derived" if has_tls else "unavailable",
                "source": "signal controller (consecutive-red streak)"},
            "estimated_density_k_veh_per_km": {"kind": "physics-derived", "source": "LW (1955) eq (2), adapted",
                "note": "OBSERVED-REGION, MIXED-REGIME density (may include both free-flow and queued vehicles "
                        "simultaneously) -- not true total-link density"},
            "observed_flow_veh_per_hour": {"kind": "physics-derived",
                "source": "LW (1955) eq (3) identity q=k*v",
                "note": "an AGGREGATE, mixed-regime flow, NOT the clean upstream arrival flow LW section 6's "
                        "shock formula would require -- kept as a general traffic-state feature only"},
            "estimated_queue_front_propagation_m_per_s": {"kind": "physics-derived (empirical)",
                "source": "not paper-sourced -- direct kinematic measurement",
                "note": "rate of change of visible_queue_length_m; positive = queue growing upstream. Valid "
                        "ONLY while queue_reaches_camera_edge is False this interval; NA when censored"},
            "estimated_hidden_queue_extension_m": {"kind": "physics-derived",
                "source": "first-order held-rate extrapolation, inspired by LW (1955) eq (17)'s shock-position "
                          "concept but not that exact solution",
                "note": "FIRST-ORDER approximation only -- extrapolates the last pre-censoring propagation rate "
                        "forward across red_duration_s while censored. A MODEL ESTIMATE, never a substitute for "
                        "the true label"},
            "q_u_cheng_veh_per_hour": {"kind": "physics-derived (Cheng et al. TRB 2010, Eq. 12)",
                "source": "Cheng, Qin, Jin & Ran (TRB 2010), Eq.(12): q_u = k_j * L_CP2 / (T_CP2 - T_r)",
                "note": "Upstream arrival-flow estimate from ONE probe's own Type II critical-point position "
                        "(L_CP2) and timestamp (T_CP2), read directly from gps_p{TAG}_probe_trajectories.csv, "
                        "using ASTRID's real signal-controller red-start time as T_r. Available on the grid "
                        "from that CP's own timestamp onward within the same red-phase cycle only; NA before "
                        "any probe has produced a resolvable Type II CP this cycle, or if no probe ever joins "
                        "the queue that cycle (a real, expected outcome on ASTRID's low-demand/OOD/no-queue "
                        "scenarios, not an error)."},
            "k_cp1_cheng_veh_per_km": {"kind": "physics-derived (Cheng et al. TRB 2010, Eq. 9)",
                "source": "Cheng, Qin, Jin & Ran (TRB 2010), Eq.(9): k_CP1 = k_j * (L_CP2 / L_CP1)",
                "note": "Local density estimate from one probe's own Type I and Type II critical-point "
                        "positions. NA whenever a Type I CP could not be resolved for that probe/cycle -- e.g. "
                        "the probe entered the observation window already mid-deceleration, or no candidate CP "
                        "passed the paper's own validation check (Methodology, Critical Points Filter step b). "
                        "This is expected on some ASTRID probes/cycles and is left NA rather than approximated."},
            "max_queue_length_cheng_m": {"kind": "physics-derived (Cheng et al. TRB 2010, Eq. 11)",
                "source": "Cheng, Qin, Jin & Ran (TRB 2010), Eq.(11): "
                          "L_q = q_s*q_u*(T_g-T_r) / (k_j*(q_s-q_u))",
                "note": "Maximum-queue-length-in-cycle estimate; T_g taken as a probe's own Type III CP "
                        "timestamp. Requires a configured Webster saturation-flow reference (q_s) -- NA for "
                        "any scenario/edge without one, or where q_s <= q_u (physically invalid chord), or "
                        "where no probe in that cycle produced a Type III CP (e.g. it never cleared the "
                        "intersection within the observed window). Available on the grid from that probe's "
                        "Type III CP timestamp onward within the same cycle only. This is Cheng et al.'s "
                        "single-probe formula; where multiple probes resolve a CP in the same cycle, the most "
                        "recently observed value is used -- this file does not implement Eq.(13)-(17)'s "
                        "multi-CP piecewise weighting."},
            "cheng_t_cp2_s": {"kind": "diagnostic (Cheng et al. TRB 2010, Type II CP)",
                "source": "gps_p{TAG}_probe_trajectories.csv, via extract_critical_points()",
                "note": "DIAGNOSTIC feature, not a Cheng equation output: the timestamp of the most recently "
                        "resolved Type II CP (queue-joining point) on this edge/cycle. Useful for QA -- e.g. "
                        "confirming Cheng CPs are actually being produced for a given scenario -- rather than "
                        "for direct model consumption. Same causal availability as q_u_cheng_veh_per_hour."},
            "cheng_l_cp2_m": {"kind": "diagnostic (Cheng et al. TRB 2010, Type II CP)",
                "source": "gps_p{TAG}_probe_trajectories.csv, via extract_critical_points()",
                "note": "DIAGNOSTIC feature: the Type II CP's own distance-to-stopline (L_CP2) that fed "
                        "Eq.(9)/(12) above. Same causal availability as q_u_cheng_veh_per_hour."},
            "cheng_t_cp3_s": {"kind": "diagnostic (Cheng et al. TRB 2010, Type III CP)",
                "source": "gps_p{TAG}_probe_trajectories.csv, via extract_critical_points()",
                "note": "DIAGNOSTIC feature: the timestamp of the most recently resolved Type III CP "
                        "(queue-discharge acceleration point) on this edge/cycle -- the T_g used by Eq.(11) "
                        "above. NA whenever no probe has produced a Type III CP yet this cycle. Same causal "
                        "availability as max_queue_length_cheng_m (available from T_CP3 onward, never earlier)."},
        })

    manifest["_removed_features"] = {
        "visible_{vehicle_type}_frac": "camera_simulator.py does not report vehicle-type composition.",
        "entry_flow_veh_per_hour / discharge_flow_veh_per_hour": "were per-vehicle boundary-crossing counts "
            "(LW eq 1, q=n/tau); not producible from the camera's 5s snapshot output.",
        "estimated_shockwave_w_bf_m_per_s": "REMOVED in v0.4: computed flux from the camera's whole visible "
            "region, which mixes free-flow and queued vehicles and is not the clean upstream state LW section "
            "6's red-signal shock formula requires. Replaced by estimated_queue_front_propagation_m_per_s.",
    }
    manifest["_unimplemented_paper_methods"] = {
        "Richards (1956)": "full text not accessible (paywalled); no equation used.",
        "Cheng et al. (2012 JITS) Eq.(4)-(8)": "flux-chord shockwave/discharge speed -- not implemented; "
            "ASTRID's empirical estimated_queue_front_propagation_m_per_s covers the analogous role instead.",
        "Cheng et al. (2012 JITS) Eq.(13)-(17)": "initial-queue detection and n-CP piecewise queue-growth "
            "weighting across multiple Type II CPs in one cycle -- not implemented; "
            "max_queue_length_cheng_m uses at most one (the most recent) probe's Type II/III CP per cycle.",
        "Rempe, Kessler & Bogenberger (2017)": "full text not accessible (paywalled); no equation used.",
        "Kumari, Bhavya, Reddy, Gopi & Lalitha (2025, ICSCN)": "full text/abstract not accessible; no "
            "concept used.",
    }
    manifest["_labels_file_note"] = (
        "true_queue_length_m / true_queue_beyond_camera / true_queue_length_future_m live only in "
        "labels_{layer}.csv, never in features_{layer}.csv. The feature-building code path never opens "
        "ground_truth/ or raw_output/vehicle_trajectories.csv. gps_p{TAG}_probe_trajectories.csv IS read "
        "(Cheng/trajectory section only) but is itself a GPS observation, not ground truth."
    )
    return manifest


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(
    scenario_dir: Path, cfg: dict, layer: str, penetration_rate: float, horizon_s: Optional[int] = None
) -> None:
    scenario = load_scenario_metadata(scenario_dir)
    tag = f"p{int(round(penetration_rate * 100)):02d}"

    has_tls = False
    k_jam = None

    if layer == "layer1":
        camera_df = load_camera_observations(scenario_dir)
        features = build_layer1_features(camera_df)
    elif layer == "layer2":
        assembled_df = load_assembled_observations(scenario_dir, tag)
        tls_df = load_tls_state(scenario_dir)
        has_tls = tls_df is not None
        probe_traj_df = load_probe_trajectories(scenario_dir, tag)
        approach_edges = cfg["network"]["approaches"]

        k_jam = compute_k_jam(scenario, cfg["vehicle_types"], DEFAULT_EFFECTIVE_GAP_M)
        webster_ref = cfg.get("_webster_reference", {})
        saturation_flow = (
            webster_ref.get("saturation_flow_per_lane_veh_per_hour", 0) * cfg["network"]["lanes_per_approach"]
            if webster_ref else None
        )

        features = build_layer2_features(
            assembled_df, tls_df, probe_traj_df, approach_edges, k_jam, saturation_flow
        )
    else:
        raise ValueError(f"Unknown layer '{layer}' (expected 'layer1' or 'layer2')")

    _validate_no_ground_truth_leak(features, f"features_{layer}")

    labels = build_labels(scenario_dir, horizon_s)
    manifest = build_feature_manifest(layer, has_tls)
    if layer == "layer2":
        manifest["_k_jam_veh_per_km"] = k_jam
        manifest["_k_jam_assumption"] = (
            f"effective_gap_m={DEFAULT_EFFECTIVE_GAP_M} -- a project assumption, NOT confirmed against "
            f"this scenario's actual vType/car-following configuration, and not sourced from any paper."
        )
    if horizon_s:
        manifest["_future_label"] = f"true_queue_length_future_m added: shifted {horizon_s}s ahead."

    output_tag = f"{layer}_{tag}" if layer == "layer2" else layer

    out_dir = scenario_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / f"features_{output_tag}.csv", index=False)
    labels.to_csv(out_dir / f"labels_{output_tag}.csv", index=False)
    with open(out_dir / f"feature_manifest_{output_tag}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"{scenario['scenario_id']}: {output_tag} features written to {out_dir}")
    print(f"  features: {features.shape[0]} rows x {features.shape[1]} cols | labels: {labels.shape[0]} rows"
          + (f" | k_jam={k_jam} veh/km | tls_state available: {has_tls}" if layer == "layer2" else ""))


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Layer 1 or Layer 2 features + labels.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--layer", type=str, default="layer1", choices=["layer1", "layer2"])
    default_pct_str = f"{DEFAULT_PENETRATION_RATE:.0%}".replace("%", "%%")
    parser.add_argument("--penetration", type=float, default=DEFAULT_PENETRATION_RATE,
                         help=f"Which GPS observation to use (layer2 only). Default {default_pct_str} "
                              f"-- ASTRID's primary experiment.")
    parser.add_argument("--horizon", type=int, default=None,
                         help="Optional future-prediction label offset in seconds (Layer 3).")
    args = parser.parse_args()

    cfg = load_network_config()

    if args.scenario:
        scenario_dirs = [SCENARIOS_DIR / args.scenario]
        if not scenario_dirs[0].exists():
            print(f"ERROR: scenario not found: {scenario_dirs[0]}")
            sys.exit(1)
    else:
        scenario_dirs = find_scenarios()
        if not scenario_dirs:
            print(f"ERROR: no scenarios found in {SCENARIOS_DIR}")
            sys.exit(1)

    failed = []
    for scenario_dir in scenario_dirs:
        try:
            process_scenario(scenario_dir, cfg, args.layer, args.penetration, args.horizon)
        except Exception as exc:
            print(f"FAILED: {scenario_dir.name}: {exc}")
            failed.append(scenario_dir.name)

    print(f"\nDone. {len(scenario_dirs) - len(failed)}/{len(scenario_dirs)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()