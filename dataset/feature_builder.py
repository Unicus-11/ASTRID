"""
feature_builder.py
====================
ASTRID Prototype -- Feature Builder, LAYER 1 + LAYER 2

LAYER 1 (unchanged from before): camera-only state estimation.
    A. what's visible   B. how it's changing   C. vehicle mix

LAYER 2 (new): adds GPS/probe features + physics-derived features +
signal-phase features on top of everything Layer 1 already builds.

--------------------------------------------------------------------------
WHERE THE PHYSICS COMES FROM (read this before trusting any physics column)
--------------------------------------------------------------------------
Lighthill, M. J. & Whitham, G. B. (1955) "On kinematic waves II: A theory
of traffic flow on long crowded roads", Proc. Roy. Soc. A 229.
  - flow q = n/tau (count of vehicles crossing a point per unit time) -- §2, eq (1).
    Used directly for entry_flow_veh_per_hour / discharge_flow_veh_per_hour below:
    both are literally vehicle counts crossing a fixed point (the 150m camera
    boundary, and the stop line) per interval.
  - density k = (sum of vehicle-seconds on a slice) / (tau * dx) -- §2, eq (2).
    Simplified here to an instantaneous count/length snapshot (their own
    eq (2) reduces to exactly this for a single-instant slice).
  - wave speed c = dq/dk, and shock (discontinuous wave) speed = the SLOPE OF
    THE CHORD joining two (k, q) points, Δq/Δk -- §2 eq (6)/(7), §3.
  - the specific "sudden stoppage" case (a red light): "It sends a shock wave
    back into the oncoming stream, at which the flow is reduced to zero and
    the concentration increased to approximately k_j... The speed of the
    shock wave is the slope of the chord... which joins the point
    representing the oncoming flow to the point (k_j, 0)." -- §6, p.338.
    This is EXACTLY estimated_shockwave_w_bf_m_per_s below: w = (0-q_A)/(k_jam-k_A).
  - jam headway (1/k_j) was empirically observed by the authors to be "only
    just greater than the average vehicle length" (~17 ft in Britain) -- p.322.
    This directly supports approximating k_jam as 1000/(avg_vehicle_length +
    min_gap) below, rather than treating it as an arbitrary assumption.
  - LW's own eq (17) gives the EXACT shock position over time for the red-signal
    case; estimated_hidden_queue_extension_m below uses the simpler first-order
    "distance = |speed| x elapsed time" approximation of the same physical
    picture, not their exact integral -- flagged as a simplification, not a
    claim of matching eq (17) precisely.

Richards, P. I. (1956) "Shock Waves on the Highway", Operations Research 4(1).
  - Independently derived the same fluid/shockwave theory the same year,
    specifically applied to signalized intersections, and found a "threshold
    effect": disturbances are minor for light traffic but build suddenly past
    a critical density -- consistent with, and cross-confirming, the LW §6
    result used here. (Referenced for context; equations used in code are
    LW's, since that is the source document actually provided.)

Cheng, Y., Qin, X., Jin, J., Ran, B. (2010/2012) "An exploratory shockwave
approach to estimating queue length using probe trajectories."
  - Their method does NOT just take the furthest-back probe seen. It finds a
    "critical point" (CP) in EACH probe's own trajectory: the specific instant
    that probe's speed drops sharply, marking the moment it joined the back
    of the queue. That position is direct empirical evidence of where the
    shockwave is, independent of the density-based LW estimate above.
    detect_probe_critical_points() below implements exactly this idea (a
    simplified version -- their full method also handles acceleration CPs
    for queue clearance, which is not implemented here yet).

NOTE ON OTHER CITED PAPERS: Rempe et al. and Lalitha et al. (2025, ICSCN)
were named as references but their full text was not available to ground
specific formulas against, so nothing here is attributed to them. The general
family of probe-based queue estimation they belong to (statistical/likelihood
methods, and hybrid deep learning temporal prediction respectively) is real
and well-established, but no equation below is claimed to come from those
specific papers. If you can share their text or a DOI, the grounding can be
tightened.
--------------------------------------------------------------------------

Structural leakage guarantee UNCHANGED: ground_truth/ is read in exactly
ONE function (build_labels), which writes to a separate labels file.

Reads (per scenario):
    scenario.json
    observations/camera_timeseries.csv           (Layer 1)
    observations/observation_p{tag}.csv           (Layer 2 -- camera+GPS merged)
    observations/gps_p{tag}_probe_ids.json        (Layer 2 -- for critical-point detection)
    raw_output/vehicle_trajectories.csv           (vehicle-mix, physics, critical points)
    raw_output/tls_state.csv                      (Layer 2 signal features, OPTIONAL)
    ground_truth/state_timeseries.csv             (ONLY inside build_labels)

Writes:
    features/features_{layer}.csv
    features/labels_{layer}.csv
    features/feature_manifest_{layer}.json

Run:
    python dataset/feature_builder.py --scenario scenario_0001 --layer layer1
    python dataset/feature_builder.py --scenario scenario_0001 --layer layer2 --penetration 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from trajectory_utils import (
    SAMPLING_INTERVAL_S,
    QUEUE_SPEED_THRESHOLD_MPS,
    load_trajectories,
    load_lane_metadata,
    attach_distance_to_stopline,
    flag_queued,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

DELTA_WINDOW_S = 30
DEFAULT_MIN_GAP_M = 2.5  # SUMO default assumption -- see compute_k_jam docstring

# Phase -> approach-group mapping, per the cross-referenced (not independently
# verified against raw <connection> data) resolution from the normal_controller.py
# audit: phases {0,2}=EW (edges 1i,2i), phases {4,6}=NS (edges 3i,4i).
PHASE_GREEN_GROUP = {0: "EW", 1: "EW", 2: "EW", 3: "EW", 4: "NS", 5: "NS", 6: "NS", 7: "NS"}
PHASE_IS_GREEN = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False, 6: True, 7: False}
GROUP_EDGES = {"EW": ["1i", "2i"], "NS": ["3i", "4i"]}
EDGE_GROUP = {e: g for g, edges in GROUP_EDGES.items() for e in edges}


def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# LAYER 1 -- unchanged
# ============================================================================

def add_change_features(obs_df: pd.DataFrame) -> pd.DataFrame:
    df = obs_df.sort_values(["approach_edge", "timestamp"]).copy()
    delta_steps = max(1, DELTA_WINDOW_S // SAMPLING_INTERVAL_S)
    g = df.groupby("approach_edge")
    df[f"queue_length_change_{DELTA_WINDOW_S}s"] = g["visible_queue_length_m"].diff(delta_steps)
    df[f"speed_change_{DELTA_WINDOW_S}s"] = g["visible_mean_speed_mps"].diff(delta_steps)
    return df


def compute_visible_composition(
    df: pd.DataFrame, approach_edges: List[str], vehicle_types: List[str],
    camera_range_m: float, sim_begin: int, sim_end: int,
) -> pd.DataFrame:
    visible = df[df["is_on_approach"] & (df["distance_to_stopline_m"] <= camera_range_m)]
    sample_times = range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S)
    rows = []
    for t in sample_times:
        snap = visible[visible["timestamp"] == t]
        for edge in approach_edges:
            on_edge = snap[snap["edge_id"] == edge]
            total = len(on_edge)
            row = {"timestamp": t, "approach_edge": edge}
            for vt in vehicle_types:
                row[f"visible_{vt}_frac"] = (on_edge["vehicle_type"] == vt).sum() / total if total > 0 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# LAYER 2 -- GPS / physics / signal
# ============================================================================

def load_observation(scenario_dir: Path, tag: str) -> pd.DataFrame:
    path = scenario_dir / "observations" / f"observation_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} -- run sensors/gps_simulator.py --penetration ... "
            f"and sensors/observation_builder.py --penetration ... first."
        )
    return pd.read_csv(path)


def load_probe_ids(scenario_dir: Path, tag: str) -> Set[str]:
    path = scenario_dir / "observations" / f"gps_{tag}_probe_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run sensors/gps_simulator.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f)["probe_vehicle_ids"])


def compute_camera_boundary_flows(
    df: pd.DataFrame, approach_edges: List[str], camera_range_m: float, sim_begin: int, sim_end: int,
) -> pd.DataFrame:
    """LW §2 eq (1): flow q = n/tau, a vehicle count crossing a fixed point per
    unit time. Applied at two boundaries: the 150m camera edge (entry_flow,
    a real-world camera could timestamp a vehicle entering its field of view)
    and the stop line, distance=0 (discharge_flow, equally observable -- the
    stop line is always inside the camera's view)."""
    d = df.sort_values(["vehicle_id", "timestamp"]).copy()
    d["is_visible"] = d["is_on_approach"] & (d["distance_to_stopline_m"] <= camera_range_m)

    prev_visible = d.groupby("vehicle_id")["is_visible"].shift(1).fillna(False)
    prev_on_approach = d.groupby("vehicle_id")["is_on_approach"].shift(1).fillna(False)
    prev_edge = d.groupby("vehicle_id")["edge_id"].shift(1)

    entered_view = (~prev_visible) & d["is_visible"]
    left_approach = prev_on_approach & (~d["is_on_approach"])

    entry_events = d.loc[entered_view, ["timestamp"]].copy()
    entry_events["approach_edge"] = d.loc[entered_view, "edge_id"]
    entry_events = entry_events[entry_events["approach_edge"].isin(approach_edges)]

    discharge_events = d.loc[left_approach, ["timestamp"]].copy()
    discharge_events["approach_edge"] = prev_edge[left_approach]
    discharge_events = discharge_events[discharge_events["approach_edge"].isin(approach_edges)]

    interval_hours = SAMPLING_INTERVAL_S / 3600.0
    sample_times = range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S)
    rows = []
    for t in sample_times:
        window_start = max(sim_begin, t - SAMPLING_INTERVAL_S)
        e_win = entry_events[(entry_events["timestamp"] > window_start) & (entry_events["timestamp"] <= t)]
        d_win = discharge_events[(discharge_events["timestamp"] > window_start) & (discharge_events["timestamp"] <= t)]
        for edge in approach_edges:
            rows.append({
                "timestamp": t,
                "approach_edge": edge,
                "entry_flow_veh_per_hour": round(len(e_win[e_win["approach_edge"] == edge]) / interval_hours, 2),
                "discharge_flow_veh_per_hour": round(len(d_win[d_win["approach_edge"] == edge]) / interval_hours, 2),
            })
    return pd.DataFrame(rows)


def compute_k_jam(scenario: dict, vehicle_types_cfg: Dict[str, dict], min_gap_m: float) -> float:
    """Jam density k_jam [veh/km]. LW (p.322) observed jam headway to be
    'only just greater than the average vehicle length' -- so
    k_jam = 1000 / (avg_vehicle_length_m + min_gap_m) is a direct application
    of their own empirical finding, not an arbitrary formula. min_gap_m is a
    SUMO-default assumption (2.5m), NOT read from your actual vtype config --
    confirm against it if precision here matters."""
    composition = scenario["vehicle_composition"]
    avg_length_m = sum(composition[vt] * vehicle_types_cfg[vt]["length"] for vt in composition)
    return round(1000.0 / (avg_length_m + min_gap_m), 2)


def detect_probe_critical_points(df: pd.DataFrame, probe_ids: Set[str], approach_edges: List[str]) -> pd.DataFrame:
    """Cheng, Qin, Jin & Ran (2012), J. Intelligent Transportation Systems 16(1).
    Their Type II CP: the instant a probe's speed drops to/below the queue
    threshold -- the moment it joined the back of the queue (their eq. 3,
    c_v,stop). Implemented below unchanged from the earlier version.

    Their Type III CP: the instant a stopped probe begins accelerating again
    -- the moment the queue starts discharging (their §8). NEW here: gives a
    second, independent empirical shockwave sighting on the discharge side,
    complementing discharge_flow_veh_per_hour.

    NOT implemented: their Type I CP (start of signal-induced deceleration)
    and the kinematic T*/L* corrections (their eq. 5-10). Cheng et al. needed
    Type I specifically to INFER the red-start time T_r from vehicle behavior,
    because they had no signal-controller access. ASTRID has real signal
    state (tls_state.csv), so T_r is read directly -- see add_signal_features's
    red_duration_s -- and Type I's whole purpose doesn't apply here. The
    kinematic corrections are a documented, scoped-out precision refinement,
    not implemented in this first pass."""
    p = df[df["vehicle_id"].isin(probe_ids) & df["is_on_approach"]].sort_values(["vehicle_id", "timestamp"]).copy()
    prev_speed = p.groupby("vehicle_id")["speed_mps"].shift(1)

    # Type II: was moving, now at/under the queue threshold.
    was_moving = prev_speed > QUEUE_SPEED_THRESHOLD_MPS
    now_queued = p["speed_mps"] <= QUEUE_SPEED_THRESHOLD_MPS
    is_type2 = was_moving.fillna(False) & now_queued

    # Type III: was queued, now moving again -- mirror image of Type II.
    was_queued = prev_speed <= QUEUE_SPEED_THRESHOLD_MPS
    now_moving = p["speed_mps"] > QUEUE_SPEED_THRESHOLD_MPS
    is_type3 = was_queued.fillna(False) & now_moving

    def _extract(mask: pd.Series, cp_type: int) -> pd.DataFrame:
        e = p.loc[mask, ["timestamp", "edge_id", "distance_to_stopline_m"]].copy()
        e = e.rename(columns={"edge_id": "approach_edge"})
        e["cp_type"] = cp_type
        return e[e["approach_edge"].isin(approach_edges)]

    return pd.concat([_extract(is_type2, 2), _extract(is_type3, 3)], ignore_index=True)


def build_probe_cp_feature(cp_events: pd.DataFrame, approach_edges: List[str], sim_begin: int, sim_end: int) -> pd.DataFrame:
    sample_times = range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S)
    rows = []
    for t in sample_times:
        window_start = max(sim_begin, t - SAMPLING_INTERVAL_S)
        window = cp_events[(cp_events["timestamp"] > window_start) & (cp_events["timestamp"] <= t)]
        for edge in approach_edges:
            edge_events = window[window["approach_edge"] == edge]
            type2 = edge_events[edge_events["cp_type"] == 2]
            type3 = edge_events[edge_events["cp_type"] == 3]
            rows.append({
                "timestamp": t,
                "approach_edge": edge,
                "probe_critical_point_count": len(type2),  # kept for backward compatibility
                "probe_shockwave_sighting_m": float(type2["distance_to_stopline_m"].max()) if len(type2) else None,
                "probe_discharge_cp_count": len(type3),
                "probe_discharge_sighting_m": float(type3["distance_to_stopline_m"].max()) if len(type3) else None,
            })
    return pd.DataFrame(rows)


def load_tls_state(scenario_dir: Path) -> Optional[pd.DataFrame]:
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
        print("  NOTE: no raw_output/tls_state.csv -- signal columns left NaN. "
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

    # BUG FIX vs the earlier version: phase_elapsed_s only measures time since
    # the last raw phase-INDEX change, not since this approach's red actually
    # began. An approach's red can span several consecutive phase indices
    # (e.g. NS/4i is red through phases 0,1,2,3, not just phase 0) -- using
    # phase_elapsed_s there under-counts how long the queue has really been
    # growing. red_duration_s below is a proper consecutive-red-streak count,
    # same run-length pattern used in trajectory_utils.flag_queued, computed
    # per approach_edge, and is what should feed the physics formulas.
    df = df.sort_values(["approach_edge", "timestamp"]).reset_index(drop=True)
    is_red = (df["is_green_for_approach"] == False)  # noqa: E712 -- NA-safe: NA==False -> False
    group_break = (~is_red).groupby(df["approach_edge"]).cumsum()
    streak_start = df.groupby(["approach_edge", group_break])["timestamp"].transform("min")
    df.loc[is_red, "red_duration_s"] = df.loc[is_red, "timestamp"] - streak_start[is_red]

    return df


def add_physics_features(df: pd.DataFrame, k_jam: float, camera_range_m: float, saturation_flow_veh_per_hour: Optional[float] = None) -> pd.DataFrame:
    """LW §2/§6 -- see module docstring for exact equation citations."""
    df = df.copy()
    camera_range_km = camera_range_m / 1000.0

    # k_A: LW eq (2), simplified to an instantaneous count/length snapshot.
    df["estimated_density_k_veh_per_km"] = df["visible_vehicle_count"] / camera_range_km

    # w_bf: LW §6 p.338 -- chord from (k_A, q_A) to (k_jam, 0). q_A approximated
    # by entry_flow (LW eq 1, a real vehicle count crossing the 150m boundary).
    # Identical in form to Cheng et al.'s v_form (their eq. 8).
    denom = k_jam - df["estimated_density_k_veh_per_km"]
    safe = denom > 1e-6
    df["estimated_shockwave_w_bf_m_per_s"] = pd.NA
    df.loc[safe, "estimated_shockwave_w_bf_m_per_s"] = (
        -(df.loc[safe, "entry_flow_veh_per_hour"] / 3600.0) / denom[safe] * 1000.0
    )

    # Hidden-queue extension: first-order approximation of LW's exact eq (17)
    # shock-position solution -- |w_bf| x time spent red. FIXED: now uses
    # red_duration_s (the actual consecutive-red streak) instead of the
    # earlier phase_elapsed_s, which under-counted red duration whenever an
    # approach's red spanned more than one raw phase index.
    df["estimated_hidden_queue_extension_m"] = pd.NA
    if "is_green_for_approach" in df.columns and "red_duration_s" in df.columns:
        is_red = df["is_green_for_approach"] == False  # noqa: E712
        has_w = df["estimated_shockwave_w_bf_m_per_s"].notna()
        has_red_duration = df["red_duration_s"].notna()
        mask = is_red & has_w & has_red_duration
        df.loc[mask, "estimated_hidden_queue_extension_m"] = (
            df.loc[mask, "estimated_shockwave_w_bf_m_per_s"].abs() * df.loc[mask, "red_duration_s"]
        )

    # -- Second, independent estimate: Cheng et al. (2012) eq (11)/(12) --
    # Their eq (12) arrival-flow formula, as summarized to me, was
    # q_u = L_CP2 / (k_jam*(T_CP2-T_r)) -- but that doesn't dimension out to
    # veh/h, and it implies a LONGER observed queue in the same time means
    # FEWER arrivals, which is backwards. Re-derived it directly from
    # conservation instead: vehicles that arrived during the red streak
    # (q_u * elapsed_time) must equal vehicles packed into the observed
    # queue at jam density (k_jam * L_CP2), so:
    #     q_u = k_jam * L_CP2 / elapsed_time
    # This is the corrected form actually used below. Their eq (11) queue-
    # length formula (L_q = q_s*q_u*(T_g-T_r) / (k_jam*(q_s-q_u))) is used
    # as given -- I checked it separately and it dimensions out correctly.
    # ADAPTED for real-time use: Cheng et al. computed this AFTER a full
    # cycle (T_g known). We use the RUNNING red duration so far in place of
    # (T_g-T_r), giving a current (not final-cycle) estimate. We also don't
    # need their Type-I-based T_r inference (see detect_probe_critical_points
    # docstring) since T_r comes directly from red_duration_s's streak start.
    df["estimated_queue_length_cheng_m"] = pd.NA
    if saturation_flow_veh_per_hour and "probe_shockwave_sighting_m" in df.columns and "red_duration_s" in df.columns:
        has_sighting = df["probe_shockwave_sighting_m"].notna()
        has_red = df["red_duration_s"].notna() & (df["red_duration_s"] > 0)
        mask = has_sighting & has_red
        if mask.any():
            q_s = saturation_flow_veh_per_hour
            L_cp2_km = df.loc[mask, "probe_shockwave_sighting_m"] / 1000.0
            T_elapsed_h = df.loc[mask, "red_duration_s"] / 3600.0
            q_u = k_jam * L_cp2_km / T_elapsed_h  # veh/h -- corrected conservation form
            q_u_safe = q_u.clip(upper=q_s * 0.98)  # avoid the eq (11) singularity as q_u -> q_s
            df.loc[mask, "estimated_queue_length_cheng_m"] = (
                q_s * q_u_safe * T_elapsed_h / (k_jam * (q_s - q_u_safe)) * 1000.0
            )

    return df


# ============================================================================
# Labels -- ground truth is read ONLY here, and only here (both layers)
# ============================================================================

def build_labels(scenario_dir: Path, horizon_s: int = None) -> pd.DataFrame:
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

def build_feature_manifest(vehicle_types: List[str], layer: str, has_tls: bool) -> dict:
    manifest = {
        "layer": layer,
        "timestamp": {"kind": "index"},
        "approach_edge": {"kind": "index"},
        "visible_vehicle_count": {"kind": "observed", "source": "camera"},
        "visible_mean_speed_mps": {"kind": "observed", "source": "camera"},
        "visible_queue_count": {"kind": "observed", "source": "camera"},
        "visible_queue_length_m": {"kind": "observed", "source": "camera"},
        "visible_occupancy_fraction": {"kind": "derived", "source": "camera"},
        "queue_reaches_camera_edge": {"kind": "observed", "source": "camera"},
        f"queue_length_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "camera history (past only)"},
        f"speed_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "camera history (past only)"},
    }
    for vt in vehicle_types:
        manifest[f"visible_{vt}_frac"] = {"kind": "observed", "source": "camera"}

    if layer == "layer2":
        manifest.update({
            "probe_count": {"kind": "observed", "source": "gps"},
            "probe_mean_speed_mps": {"kind": "observed", "source": "gps"},
            "probe_min_distance_to_stopline_m": {"kind": "observed", "source": "gps"},
            "probe_max_distance_to_stopline_m": {"kind": "observed", "source": "gps",
                "leakage_risk": "none -- a real sighting, may exceed camera_range_m by design"},
            "probe_critical_point_count": {"kind": "derived", "source": "gps (Cheng et al. 2012)"},
            "probe_shockwave_sighting_m": {"kind": "derived", "source": "gps (Cheng et al. 2012)",
                "note": "position of the most recent probe deceleration critical point -- an empirical, "
                        "not modelled, shockwave sighting"},
            "entry_flow_veh_per_hour": {"kind": "derived", "source": "camera boundary crossing (LW eq 1)"},
            "discharge_flow_veh_per_hour": {"kind": "derived", "source": "camera boundary crossing (LW eq 1)"},
            "estimated_density_k_veh_per_km": {"kind": "physics-derived", "source": "LW eq (2), simplified"},
            "estimated_shockwave_w_bf_m_per_s": {"kind": "physics-derived", "source": "LW section 6 (chord to k_jam,0)",
                "note": "a MODEL ESTIMATE, not ground truth"},
            "estimated_hidden_queue_extension_m": {"kind": "physics-derived",
                "source": "first-order approximation of LW eq (17)",
                "note": "a MODEL ESTIMATE -- never a substitute for the true_hidden_queue_m-style label"},
            "current_phase": {"kind": "observed" if has_tls else "unavailable", "source": "signal controller"},
            "phase_elapsed_s": {"kind": "observed" if has_tls else "unavailable", "source": "signal controller"},
            "is_green_for_approach": {"kind": "derived" if has_tls else "unavailable",
                "note": "inherits the not-independently-verified phase-to-approach mapping caveat"},
        })

    manifest["_labels_file_note"] = (
        "true_queue_length_m etc. live in labels_{layer}.csv, never in features_{layer}.csv. "
        "The feature-building code path never opens ground_truth/."
    )
    return manifest


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict, layer: str, penetration_rate: float, horizon_s: int = None) -> None:
    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    camera_range_m = cfg["network"]["camera_range_m"]
    vehicle_types = list(cfg["vehicle_types"].keys())
    sim_begin, sim_end = int(scenario["simulation_begin"]), int(scenario["simulation_end"])
    tag = f"p{int(round(penetration_rate * 100)):02d}"

    if layer == "layer1":
        camera_path = scenario_dir / "observations" / "camera_timeseries.csv"
        if not camera_path.exists():
            raise FileNotFoundError(f"Missing {camera_path} -- run sensors/camera_simulator.py first.")
        obs = pd.read_csv(camera_path)
    elif layer == "layer2":
        obs = load_observation(scenario_dir, tag)
    else:
        raise ValueError(f"Unknown layer '{layer}' (expected 'layer1' or 'layer2')")

    features = add_change_features(obs)
    features["visible_occupancy_fraction"] = (features["visible_queue_length_m"] / camera_range_m).clip(upper=1.0)

    raw = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()
    raw = attach_distance_to_stopline(raw, lane_metadata, approach_edges)
    raw = flag_queued(raw)

    composition_df = compute_visible_composition(raw, approach_edges, vehicle_types, camera_range_m, sim_begin, sim_end)
    features = features.merge(composition_df, on=["timestamp", "approach_edge"], how="left")

    has_tls = False
    if layer == "layer2":
        flows_df = compute_camera_boundary_flows(raw, approach_edges, camera_range_m, sim_begin, sim_end)
        features = features.merge(flows_df, on=["timestamp", "approach_edge"], how="left")

        probe_ids = load_probe_ids(scenario_dir, tag)
        cp_events = detect_probe_critical_points(raw, probe_ids, approach_edges)
        cp_df = build_probe_cp_feature(cp_events, approach_edges, sim_begin, sim_end)
        features = features.merge(cp_df, on=["timestamp", "approach_edge"], how="left")

        tls_df = load_tls_state(scenario_dir)
        has_tls = tls_df is not None
        features = add_signal_features(features, tls_df)

        k_jam = compute_k_jam(scenario, cfg["vehicle_types"], DEFAULT_MIN_GAP_M)
        # Per-approach saturation flow: reuses the SAME saturation_flow_per_lane
        # constant normal_controller.py's Webster calculation already relies on
        # (see scenario_config.json's _webster_reference), scaled by this
        # approach's own lane count rather than the combined NS/EW group total.
        webster_ref = cfg.get("_webster_reference", {})
        saturation_flow = (
            webster_ref.get("saturation_flow_per_lane_veh_per_hour", 0) * cfg["network"]["lanes_per_approach"]
            if webster_ref else None
        )
        features = add_physics_features(features, k_jam, camera_range_m, saturation_flow)

    labels = build_labels(scenario_dir, horizon_s)
    manifest = build_feature_manifest(vehicle_types, layer, has_tls)
    if layer == "layer2":
        manifest["_k_jam_veh_per_km"] = k_jam
        manifest["_k_jam_assumption"] = f"min_gap_m={DEFAULT_MIN_GAP_M} (SUMO default, not confirmed against your vtype config)"
    if horizon_s:
        manifest["_future_label"] = f"true_queue_length_future_m added: shifted {horizon_s}s ahead."

    # Layer 2 output is tagged with the penetration rate so that running with
    # different --penetration values produces separate files (features_layer2_p05.csv,
    # features_layer2_p10.csv, ...) instead of overwriting one generic file.
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
    parser.add_argument("--penetration", type=float, default=0.10,
                         help="Which GPS observation to use (layer2 only).")
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