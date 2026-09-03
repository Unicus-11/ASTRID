"""
feature_builder.py
====================
ASTRID Prototype -- Feature Builder, LAYER 1 + LAYER 2

LAYER 1: camera-only state estimation.
    Directly observed camera fields + history-derived (past-only) deltas
    computed from those same fields.

LAYER 2: adds GPS/probe observations + signal-phase features + physics-
    derived features on top of everything Layer 1 already builds.

--------------------------------------------------------------------------
v0.5 -- LITERATURE VERIFICATION PASS (this revision)
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
  publication) -- WAS obtained in full text and read directly. That gives
  a primary-source basis for Cheng's method (see the dedicated section
  below), and confirms decisively that the method requires PER-VEHICLE,
  PER-SECOND trajectory data for a specifically-identified probe -- data
  ASTRID's gps_simulator.py does not, and by design cannot, expose (it
  reports 5-second AGGREGATE probe statistics per approach edge, not
  individual probe position/speed/acceleration traces). This is not an
  assumption -- it is read directly from their Eq.(1)-(3) (critical-point
  extraction, operating on a continuous per-vehicle trajectory) and their
  Eq.(9)-(10) and (12) (which require L_CP1, L_CP2, T_CP2 -- the exact
  position and timestamp of ONE specific probe's detected critical
  point). ASTRID's Cheng-inspired feature is discussed on that basis below.

  Rempe, Kessler & Bogenberger (2017), "Fusing Probe Speed and Flow Data
  for Robust Short-Term Congestion Front Forecasts", 5th IEEE MT-ITS,
  Naples, DOI 10.1109/MTITS.2017.8005695 -- confirmed to exist (title,
  venue, DOI, page range 31-36 verified via multiple citing works and the
  authors' own publication lists), but the paper itself is paywalled at
  IEEE Xplore and no accessible full text or abstract was found. NO
  equation or specific claim is attributed to this paper anywhere in this
  file. (A DIFFERENT, later Rempe co-authored paper -- Rempe, Loder &
  Bogenberger, "Estimating motorway traffic states with data fusion and
  physics-informed deep learning" -- was separately supplied in full text
  and is cited below purely as general motivation for the q=k*v identity
  used in this PIDL-adjacent literature; it is not the 2017 paper and is
  labeled as such wherever referenced.)

  "Lalitha et al. (2025, ICSCN)" -- tracked down via a bibliography entry
  in a supplied, unrelated YOLO/traffic-detection survey paper: the full
  reference is R. T. Kumari, M. Bhavya, J. A. Reddy, T. T. Gopi and
  R. Lalitha, "Intelligent Traffic Flow Prediction Using Hybrid Deep
  Learning Models," 2025 International Conference on Sustainable
  Communication Networks and Application (ICSCN), Theni, India, 2025,
  pp.639-644. R. Lalitha is the LAST-listed author, not the first --
  the correct short citation form is Kumari et al., not "Lalitha et al."
  No abstract, full text, or further detail for this paper was found
  anywhere searchable (it appears to be a small regional-conference paper
  not yet indexed with an abstract). NO concept, equation, or claim is
  attributed to it anywhere in this file.

Net effect on the code: LW is the only paper from which a specific
equation is directly cited. Richards, Rempe (2017), and Kumari/Lalitha
(2025) are acknowledged as named references ASTRID's project documentation
points to, but nothing in this file's logic is justified by an equation
from any of them, because their full texts were not accessible. Cheng et
al.'s method is fully understood (via the 2010 TRB precursor, same authors
/ same method) and is explicitly NOT implemented, for the sensor-schema
reason stated in its own section below -- with the one adaptation that
does exist now precisely mapped against Cheng's actual Eq.(11)/(12)
rather than a paraphrase.

--------------------------------------------------------------------------
v0.4 -- SHOCKWAVE / JAM-DENSITY CORRECTNESS FIXES (prior revision)
--------------------------------------------------------------------------
1. estimated_shockwave_w_bf_m_per_s (LW flux-chord estimate) -- REMOVED.
   It computed k_A/q_A from visible_vehicle_count / visible_mean_speed_mps
   averaged over the WHOLE camera region. The LW red-signal shock
   construction (w = (0-q_A)/(k_jam-k_A), LW section 6, p.338) requires
   (k_A, q_A) to be the clean INCOMING/upstream traffic state -- but the
   camera's 150m region frequently contains BOTH free-flow vehicles and
   the queue itself, and the camera reports one aggregate speed for the
   whole region, so it cannot isolate an upstream-only substream. This
   produces a pathological failure exactly when the feature matters most:
   as the queue lengthens, more of the visible region is queued, so
   visible_mean_speed_mps drops, so the estimated q_A drops, so the
   estimated shock speed approaches ZERO -- even while the true queue
   front may be propagating upstream rapidly. Removed rather than patched.

2. estimated_queue_front_propagation_m_per_s -- NEW, replaces the role
   estimated_shockwave_w_bf_m_per_s used to play in
   estimated_hidden_queue_extension_m. Instead of deriving a shock speed
   from flux theory applied to a mixed-regime state, this measures the
   shock/queue-front speed the way it is actually observable here: the
   rate of change of visible_queue_length_m over DELTA_WINDOW_S. Valid
   ONLY while the queue's back edge is still inside the camera's view
   (queue_reaches_camera_edge == False this row); NA when censored, since
   the true front position is no longer observable then. Not attributed
   to any paper -- it is a plain kinematic measurement of an observed
   boundary's motion, not a flux-theory result.

3. estimated_hidden_queue_extension_m -- REWORKED to use (2). A "hidden
   extension" is only meaningful once the queue is actually CENSORED --
   exactly the period (2) cannot measure directly, since the front is no
   longer visible. Fix: forward-fill the LAST OBSERVED, PRE-CENSORING
   propagation rate per approach_edge, hold it constant, and extrapolate
   forward using red_duration_s once censoring begins. Explicitly a
   first-order, held-rate extrapolation -- not LW's exact eq (17) shock-
   position solution. Rate clipped at zero before extrapolating (assumes
   the queue does not shrink while still red and still censored).

4. compute_k_jam() docstring separates LITERATURE from ASTRID'S OWN
   MODELING ASSUMPTION. LW supports the qualitative claim "jam headway is
   related to vehicle length" (p.322) -- it does NOT give the specific
   formula k_jam = 1000/(avg_vehicle_length_m + effective_gap_m). That
   formula is ASTRID's own project-level assumption, built in the spirit
   of LW's finding but not derived from their equations.

5. DEFAULT_MIN_GAP_M renamed DEFAULT_EFFECTIVE_GAP_M and no longer called
   "the SUMO default" -- SUMO's car-following model has several distinct
   gap-related parameters, and this value was never actually checked
   against this project's scenario_config.json vType definitions.
   Documented as an unverified project assumption.

6. _safe_divide() bug fix: previously accepted abs(denominator) >
   min_denominator, including negative values. Every physical use in this
   file (camera_range_m, elapsed time, a density/saturation-flow gap that
   must be positive to be physically meaningful) requires a STRICTLY
   POSITIVE denominator. Fixed to require denominator > min_denominator.

--------------------------------------------------------------------------
v0.3 -- SENSOR-REALISM REWRITE (prior revision, summary retained)
--------------------------------------------------------------------------
raw_output/vehicle_trajectories.csv is not read anywhere in this file.
Removed (as not observable by ASTRID's actual sensor outputs):
    - visible_{vehicle_type}_frac (camera reports no vehicle-type field)
    - entry_flow_veh_per_hour / discharge_flow_veh_per_hour (required
      per-second boundary-crossing events; camera reports a 5s snapshot)
    - probe_critical_point_count / probe_shockwave_sighting_m /
      probe_discharge_cp_count / probe_discharge_sighting_m (required
      per-second speed traces for identified probes; GPS reports 5s
      AGGREGATE probe statistics only)

--------------------------------------------------------------------------
WHERE THE PHYSICS COMES FROM (read this before trusting any physics column)
--------------------------------------------------------------------------
Lighthill, M. J. & Whitham, G. B. (1955) "On kinematic waves II: A theory
of traffic flow on long crowded roads", Proc. Roy. Soc. A 229, pp.317-345.
FULL TEXT READ DIRECTLY. Section/equation numbers below refer to that text.
  - §2, eq (1): q = n/tau -- flow as a boundary-crossing vehicle count per
    unit time. NOT used in this file: the camera's 5-second snapshot does
    not report crossing events, only region occupancy + speed (see v0.3).
  - §2, eq (2): k = (sum of vehicle-crossing-times)/(tau*dx) -- density as
    a time-averaged quantity over a road slice. estimated_density_k_veh_per_km
    below is an ADAPTATION: an instantaneous count/length snapshot over
    the camera's own 150m visible region, not LW's time-averaged slice
    quantity, and not a true total-link density (see point 7 below).
  - §2, eq (3): v = q/k, the space-mean speed -- algebraically this is the
    same relationship as q = k*v. observed_flow_veh_per_hour below uses
    this identity, fed with ASTRID's own density/speed approximations
    (not LW's rigorously time-averaged q and k), and is explicitly an
    AGGREGATE, MIXED-REGIME estimate over whatever mix of free-flow and
    queued vehicles the camera currently sees -- not LW's clean single-
    regime q(k) functional relationship.
  - §2 eq (6)/(7), §3: wave speed c = dq/dk; shock (discontinuous wave)
    speed = the SLOPE OF THE CHORD joining two (k,q) points on the flow-
    concentration curve, Δq/Δk. This chord construction is theoretically
    correct for the red-light shock, but REQUIRES a clean upstream traffic
    state as one of the two chord endpoints -- see point below.
  - §6, p.338: "It sends a shock wave back into the oncoming stream, at
    which the flow is reduced to zero and the concentration increased to
    approximately k_j... The speed of the shock wave is the slope of the
    chord... which joins the point representing the oncoming flow to the
    point (k_j, 0)." This is the LW red-signal shock construction. It is
    NOT implemented as a flux-based formula in this file (see v0.4 point 1
    above) because ASTRID's camera cannot isolate the required "oncoming
    flow" (upstream-only) state from its mixed 150m aggregate.
    estimated_queue_front_propagation_m_per_s instead measures the same
    physical quantity (the shock/queue-front speed) empirically, directly
    from the observed movement of the queue's own back edge.
  - p.322: jam headway (1/k_j) empirically observed to be "only just
    greater than the average vehicle length" (~17 ft in Britain). This is
    the LITERATURE CLAIM. See compute_k_jam() for how ASTRID's own
    modeling assumption (a specific formula) is built in that spirit but
    is not itself derived from LW's equations (v0.4 point 4).
  - §6, eq (17): the EXACT shock position over time for the red-signal
    case. estimated_hidden_queue_extension_m is a first-order, held-rate
    approximation of the same physical picture (v0.4 point 3), not their
    exact integral.

Richards, P. I. (1956) "Shock Waves on the Highway", Operations Research
4(1), pp.42-51. NOT ACCESSIBLE IN FULL TEXT (see v0.5 note above). No
equation from this paper is used or cited in this file. Referenced only
as a named part of ASTRID's project documentation, at the general,
non-equation level described in the v0.5 note.

Cheng, Y., Qin, X., Jin, J., Ran, B. -- "An Exploratory Shockwave Approach
to Estimating Queue Length Using Probe Trajectories", Journal of
Intelligent Transportation Systems 16(1), pp.12-23 (2012). The 2012 JITS
text itself was not accessible, but the same four authors' TRB 2010
conference paper on the identical method WAS read in full. Per that text:
  - A probe's trajectory is a series of (location, speed, acceleration)
    records. A "critical point" (CP) is where the vehicle's motion regime
    changes (their Eq.(1)-(3), extracted by comparing each new point's
    speed/acceleration against a moving threshold on the recent trajectory
    segment) -- this REQUIRES a continuous per-vehicle trajectory, not an
    aggregated snapshot.
  - Type II CP: the point where a vehicle slows down and joins the queue.
    Type III CP: the point where it starts accelerating again (queue
    discharge). (Type I CP -- start of signal-induced deceleration -- is
    used by Cheng et al. only to INFER the unknown red-start time; ASTRID
    has real signal-controller state, so this is moot here regardless.)
  - Their queue-formation shock speed (Eq.(8)): v_form = (0 - q_u)/(k_j - k_u),
    where q_u, k_u are the UPSTREAM ARRIVAL flow/density -- the same
    "clean upstream state" requirement as LW section 6.
  - They do not measure q_u or k_u directly; instead (Eq.(9)-(10)) they
    approximate a local density k_CP1 = k_j * (L_CP2 / L_CP1) from a
    SINGLE probe's own Type I and Type II critical-point POSITIONS
    (L_CP1, L_CP2), and re-derive an upstream arrival-flow estimate from
    that.
  - Their maximum-queue-length formula (Eq.(11)):
        L_q = q_s * q_u * (T_g - T_r) / (k_j * (q_s - q_u))
    and their arrival-flow estimate (Eq.(12)):
        q_u = k_j * L_CP2 / (T_CP2 - T_r)
    where L_CP2 is the EXACT distance-to-stopbar of ONE specific probe's
    detected Type II critical point, and T_CP2 is that same probe's
    critical-point TIMESTAMP -- both derived from that probe's continuous
    trajectory, not from a 5-second population aggregate.

  WHY THIS IS NOT IMPLEMENTED IN ASTRID: gps_simulator.py's actual output
  is a 5-second AGGREGATE across however many probes are present on an
  approach that interval (probe_count, probe_mean_speed_mps,
  probe_min/max_distance_to_stopline_m) -- there is no continuous
  per-vehicle trajectory to run CP extraction against, and no way to
  identify "the one probe whose Type II critical point this is." This is
  confirmed directly from Cheng et al.'s own method description, not
  inferred. An EARLIER version of this file's Cheng-inspired feature used
  probe_max_distance_to_stopline_m (the farthest of possibly several
  probes seen in a 5-second bin) as a substitute for their L_CP2, and
  red_duration_s as a substitute for (T_CP2 - T_r) in a
  conservation-style re-derivation that is structurally identical in FORM
  to their Eq.(11)/(12). Having now read the actual method, that
  substitution is a materially different and coarser measurement than
  what Cheng et al.'s equations require -- an aggregate 5-second maximum
  across an unknown number of probes is not the same quantity as one
  specific probe's precisely-timestamped critical-point position, and can
  overstate the queue-front position (by picking up the single farthest
  probe in the window) or lag it (probes are only sampled every 5s, so a
  probe's exact joining moment is never observed). Given the strict
  sensor-realism rule, this feature is REMOVED from this revision rather
  than kept as a labeled-but-shaky adaptation. If gps_simulator.py is ever
  intentionally upgraded to expose per-probe trajectories (position/speed
  at 1s resolution, keyed by probe id, across consecutive intervals),
  Cheng's actual Eq.(9)-(12) could be implemented as a separate, real
  baseline at that point -- not before.

Rempe, F., Kessler, L., Bogenberger, K. (2017), "Fusing Probe Speed and
Flow Data for Robust Short-Term Congestion Front Forecasts," 5th IEEE
MT-ITS. NOT ACCESSIBLE IN FULL TEXT (see v0.5 note). No equation or claim
from this paper is used in this file.

Kumari, R. T., Bhavya, M., Reddy, J. A., Gopi, T. T., Lalitha, R. (2025),
"Intelligent Traffic Flow Prediction Using Hybrid Deep Learning Models,"
ICSCN 2025, Theni, India, pp.639-644 (previously mis-cited in ASTRID
project docs as "Lalitha et al." -- R. Lalitha is the last-listed author).
NOT ACCESSIBLE IN FULL TEXT OR ABSTRACT. No concept or claim from this
paper is used in this file.
--------------------------------------------------------------------------

Structural leakage guarantee: ground_truth/ is read in exactly ONE
function (build_labels), which writes to a separate labels file.
raw_output/vehicle_trajectories.csv is not read anywhere in this file.

Reads (per scenario):
    scenario.json
    observations/camera_timeseries.csv           (Layer 1 + base of Layer 2)
    observations/assembled_observations_p{tag}.csv (Layer 2 -- from
                                                     observation_assembler.py)
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
from typing import List, Optional

import pandas as pd

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
GROUP_EDGES = {"EW": ["1i", "2i"], "NS": ["3i", "4i"]}
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

    # red_duration_s: a proper consecutive-red-streak count per approach_edge
    # (an approach's red can span several raw phase indices, so
    # phase_elapsed_s alone under-counts how long the queue has really been
    # growing). Same run-length pattern trajectory_utils.flag_queued uses.
    df = df.sort_values(["approach_edge", "timestamp"]).reset_index(drop=True)
    is_red = (df["is_green_for_approach"] == False)  # noqa: E712 -- NA-safe: NA==False -> False
    group_break = (~is_red).groupby(df["approach_edge"]).cumsum()
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

    (A Cheng-et-al.-inspired queue-length estimate was implemented in an
    earlier revision of this file, using probe_max_distance_to_stopline_m
    as a substitute for Cheng et al.'s L_CP2. Having now read Cheng et
    al.'s actual method in full (see module docstring), that substitution
    was found to be a materially coarser measurement than what their
    equations require, and the feature has been REMOVED rather than kept
    as a shaky adaptation -- see the Cheng section of the module
    docstring for the full reasoning.)
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


def build_layer2_features(
    assembled_df: pd.DataFrame,
    tls_df: Optional[pd.DataFrame],
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
        })

    manifest["_removed_features"] = {
        "visible_{vehicle_type}_frac": "camera_simulator.py does not report vehicle-type composition.",
        "entry_flow_veh_per_hour / discharge_flow_veh_per_hour": "were per-vehicle boundary-crossing counts "
            "(LW eq 1, q=n/tau); not producible from the camera's 5s snapshot output.",
        "probe_critical_point_count / probe_shockwave_sighting_m / probe_discharge_cp_count / "
            "probe_discharge_sighting_m / estimated_queue_length_cheng_inspired_m": "Cheng et al.'s method "
            "(read in full via its 2010 TRB precursor -- see module docstring) requires per-vehicle, per-second "
            "trajectory data to detect a specific probe's critical point; gps_simulator.py exposes only 5s "
            "AGGREGATE probe statistics. An earlier adaptation using the aggregate "
            "probe_max_distance_to_stopline_m field as a substitute for Cheng et al.'s L_CP2 was found, on "
            "reading their actual equations, to be a materially coarser and different quantity -- removed "
            "rather than kept as a mislabeled approximation.",
        "estimated_shockwave_w_bf_m_per_s": "REMOVED in v0.4: computed flux from the camera's whole visible "
            "region, which mixes free-flow and queued vehicles and is not the clean upstream state LW section "
            "6's red-signal shock formula requires. Replaced by estimated_queue_front_propagation_m_per_s.",
    }
    manifest["_unimplemented_paper_methods"] = {
        "Richards (1956)": "full text not accessible (paywalled); no equation used.",
        "Cheng et al. (2012)": "full method understood via its 2010 TRB precursor; not implementable with "
            "ASTRID's current 5s-aggregate GPS sensor -- requires per-probe trajectories.",
        "Rempe, Kessler & Bogenberger (2017)": "full text not accessible (paywalled); no equation used.",
        "Kumari, Bhavya, Reddy, Gopi & Lalitha (2025, ICSCN)": "full text/abstract not accessible; no "
            "concept used.",
    }
    manifest["_labels_file_note"] = (
        "true_queue_length_m / true_queue_beyond_camera / true_queue_length_future_m live only in "
        "labels_{layer}.csv, never in features_{layer}.csv. The feature-building code path never opens "
        "ground_truth/ or raw_output/vehicle_trajectories.csv."
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

        k_jam = compute_k_jam(scenario, cfg["vehicle_types"], DEFAULT_EFFECTIVE_GAP_M)
        webster_ref = cfg.get("_webster_reference", {})
        saturation_flow = (
            webster_ref.get("saturation_flow_per_lane_veh_per_hour", 0) * cfg["network"]["lanes_per_approach"]
            if webster_ref else None
        )

        features = build_layer2_features(assembled_df, tls_df, k_jam, saturation_flow)
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