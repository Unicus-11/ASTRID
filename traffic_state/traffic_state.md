============================================================
ASTRID — LAYER 1
TRAFFIC STATE ESTIMATION & QUEUE RECONSTRUCTION
============================================================


1. PURPOSE OF THIS LAYER
============================================================

Layer 1 answers:

    "What is happening on each approach right now?"

The real sensors do NOT see every vehicle.

We have:

    GPS
        -> sparse vehicle positions and speeds

    CCTV
        -> local vehicle observations

Therefore:

    observed vehicles != total vehicles

The purpose of this layer is to reconstruct the hidden
traffic state from these incomplete observations.

The main outputs are:

    - traffic density
    - traffic flow
    - average speed
    - queue state
    - estimated queue length
    - queue growth / dissipation
    - shockwave speed
    - confidence / uncertainty


------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

GPS and CCTV are NOT themselves the queue estimator.

They provide observations.

LWR + shockwave theory provides the physical traffic-flow
relationship.

Later, ML/PINN can learn to reconstruct states that are
not directly observed.

Conceptually:

    GPS + CCTV
         |
         v
    Observed State
         |
         v
    State Estimator
         |
         +------> LWR conservation
         |
         +------> Shockwave model
         |
         v
    Estimated Traffic State
         |
         v
    Queue / Future-State Features


============================================================
2. INPUT DATA
============================================================

Primary input:

    datasets/scenario_XXXX/sensor_dataset.json


Each record represents approximately one simulation second.

Example information already available:

    simulation_time

    traffic:
        north
        south
        east
        west

    For each approach:
        vehicles
        queue
        speed
        approach_arrivals

    gps_count

    cctv_count

    camera_counts

    vehicle_features

    sensors:
        ground_truth
        vehicle position
        vehicle speed
        edge
        lane
        lane_position
        movement
        vehicle_type


Ground truth is used mainly for:

    validation
    training labels
    evaluation

It should NOT be treated as if it were a real-world sensor.


============================================================
3. WHY WE NEED STATE ESTIMATION
============================================================

Suppose:

    100 vehicles are actually on an approach.

CCTV sees:

    35 vehicles

GPS sees:

    12 probe vehicles

The system cannot simply say:

    "There are 47 vehicles."

That would confuse observations with the actual state.

Instead, the system estimates:

    total traffic
    density
    flow
    queue length
    traffic regime

from the available observations.

This is the central missing-state problem of ASTRID.

The project paper explicitly identifies incomplete sensing and
sparse connected data as fundamental problems. Missing vehicles
must not be interpreted as zero vehicles.


============================================================
4. TRAFFIC VARIABLES
============================================================

For each approach we work mainly with:

    k = traffic density
        vehicles / metre

    q = traffic flow
        vehicles / second

    v = average traffic speed
        metres / second

These variables are related by:

    q = k × v


Therefore, if two of the variables are known or estimated,
the third can be obtained.


============================================================
5. LWR CONSERVATION EQUATION
============================================================

The fundamental physical constraint is:

    ∂k/∂t + ∂q/∂x = 0

Meaning:

    change in vehicles
    =
    vehicles entering
    -
    vehicles leaving

This prevents the estimator from producing physically
impossible traffic states.

LWR is therefore the PHYSICS layer.

It does not replace the sensor data.


============================================================
6. SHOCKWAVE THEORY
============================================================

Shockwaves describe how the boundary between two traffic
states moves.

For two traffic states A and B:

    w = (q_B - q_A) / (k_B - k_A)

where:

    w = shockwave speed
    q = flow
    k = density


------------------------------------------------------------
BACKWARD-FORMING SHOCKWAVE
------------------------------------------------------------

When a signal becomes red, vehicles begin accumulating.

The queue boundary normally propagates upstream.

Using:

    q_B = 0
    k_B = k_jam

we obtain:

    w_bf =
        -q_A / (k_jam - k_A)

The negative sign means the queue boundary moves
back toward the incoming traffic.


------------------------------------------------------------
RECOVERY SHOCKWAVE
------------------------------------------------------------

When the signal becomes green, the queue begins to
dissipate.

The recovery boundary can be represented as:

    w_br =
        q_C / (k_C - k_jam)


------------------------------------------------------------
QUEUE EXTENSION
------------------------------------------------------------

If a backward-forming shockwave exists for a red period:

    L_max = |w_bf| × T_red

where:

    L_max = maximum estimated queue extension
    T_red = effective red duration


------------------------------------------------------------
QUEUE CLEARANCE
------------------------------------------------------------

Approximate queue clearance time:

    T_clear =
        L_max / |w_br|


IMPORTANT:

Shockwave speed is NOT vehicle speed.

It describes movement of a TRAFFIC-STATE BOUNDARY.


============================================================
7. QUEUE ESTIMATION
============================================================

The system combines:

    CCTV observations
        +
    GPS probe observations
        +
    traffic-flow state
        +
    LWR conservation
        +
    shockwave dynamics

to estimate the queue that cannot be directly observed.

The goal is NOT simply:

    "How many vehicles does the camera see?"

The goal is:

    "How long is the actual queue?"


For example:

    CCTV observed vehicles = 25
    GPS probes = 8

The estimated queue might be:

    queue_length = 142 metres

The exact value comes from the state-estimation process,
not directly from CCTV count.


============================================================
8. FEATURES GENERATED BY LAYER 1
============================================================

YES — this layer generates features.

These features become the input to later ML/prediction layers.

For each approach we can generate:

TRAFFIC FEATURES

    density
    flow
    average_speed
    speed_variance
    arrival_rate
    departure_rate
    vehicle_count


QUEUE FEATURES

    observed_queue
    estimated_queue_length
    queue_growth_rate
    queue_dissipation_rate
    queue_duration
    queue_status


SHOCKWAVE FEATURES

    backward_shockwave_speed
    recovery_shockwave_speed
    estimated_max_queue
    estimated_clearance_time


SENSOR FEATURES

    gps_count
    cctv_count
    gps_coverage
    cctv_coverage
    sensor_agreement


TRAFFIC COMPOSITION

    bike_ratio
    car_ratio
    bus_ratio
    hgv_ratio


MOVEMENT FEATURES

    left_turn_rate
    straight_rate
    right_turn_rate


TIME FEATURES

    simulation_time
    signal_phase
    red_time_elapsed
    green_time_elapsed


CONFIDENCE

    state_confidence
    queue_confidence

Confidence becomes important because sparse sensors can
produce uncertain estimates.


============================================================
9. PROPOSED FILE STRUCTURE
============================================================

ASTRID/

    datasets/
        scenario_0001/
            sensor_dataset.json

        scenario_0002/
            sensor_dataset.json

        ...

    traffic_state/

        __init__.py

        config.py

        traffic_state.py
            Basic traffic-state calculations.

        density_estimator.py
            Estimate density from observations.

        flow_estimator.py
            Estimate traffic flow.

        queue_estimator.py
            Estimate queue length.

        shockwave.py
            LWR/shockwave calculations.

        feature_builder.py
            Convert estimated state into ML features.

        state_estimator.py
            Main Layer-1 pipeline.

        schemas.py
            Input/output data structures.

        README.txt


    outputs/

        traffic_state/
            scenario_0001/
                state_features.json

            scenario_0002/
                state_features.json

            ...


============================================================
10. INPUT → OUTPUT
============================================================

INPUT:

    sensor_dataset.json

        |
        v

    GPS observations
    CCTV observations
    traffic measurements
    signal state
    vehicle information

        |
        v

    STATE ESTIMATION

        |
        +---- density
        |
        +---- flow
        |
        +---- speed
        |
        +---- queue
        |
        +---- shockwave
        |
        +---- confidence

        |
        v

OUTPUT:

    state_features.json


The output is NOT yet the final AI prediction.

It is the reconstructed traffic state.


============================================================
11. EXAMPLE OUTPUT
============================================================

A simplified record could look like:

{
    "scenario": "scenario_0029",
    "simulation_time": 1200,

    "north": {
        "density": ...,
        "flow": ...,
        "speed": ...,
        "queue_length": ...,
        "queue_growth_rate": ...,
        "shockwave_speed": ...,
        "clearance_time": ...,
        "confidence": ...
    },

    "south": {
        ...
    },

    "east": {
        ...
    },

    "west": {
        ...
    }
}


============================================================
12. WHAT HAPPENS TO THIS OUTPUT?
============================================================

Layer 1 output
      |
      v
Traffic state features
      |
      v
Layer 2 — Traffic Regime / Prediction
      |
      v
Future arrivals + turning behaviour
      |
      v
Layer 3 — Multi-intersection / Spatial Model
      |
      v
Layer 4 — Signal Optimization
      |
      v
PPO / MARL
      |
      v
Safety Supervisor
      |
      v
Signal Controller


============================================================
13. IMPORTANT: CURRENT PROTOTYPE LIMITATION
============================================================

The final ASTRID architecture is designed for multiple
connected intersections.

The complete architecture includes:

    sparse GPS
        +
    CCTV
        +
    state reconstruction
        +
    LWR/shockwaves
        +
    ST-GCN/PINN
        +
    future prediction
        +
    multi-agent coordination
        +
    PPO/MARL
        +
    safety layer

However, the current prototype has only ONE junction.

Therefore we should NOT pretend to implement the
multi-intersection part now.

For the current prototype:

    ONE JUNCTION
        |
        v
    FOUR APPROACHES
        |
        v
    STATE ESTIMATION
        |
        v
    QUEUE / SHOCKWAVE ANALYSIS


Later, when multiple junctions exist:

    Junction A
        |
        +------ Junction B
        |
        +------ Junction C

ST-GCN / spatial-temporal models and MARL become useful.


============================================================
14. REMAINING ASTRID LAYERS — OVERVIEW
============================================================

LAYER 1
Traffic State Estimation

    What is happening now?

    GPS + CCTV
        ↓
    state reconstruction
        ↓
    LWR + shockwaves
        ↓
    queue / traffic features


LAYER 2
Traffic Prediction

    What will happen next?

    Use historical state + current state
        ↓
    predict:
        arrivals
        turning behaviour
        queue evolution


LAYER 3
Network / Multi-Junction Intelligence

    What will neighbouring junctions experience?

    Requires multiple connected junctions.

    ST-GCN / spatial-temporal modelling
        +
    predicted traffic states


LAYER 4
Signal Optimization

    What should the signal do?

    Candidate actions
        ↓
    PPO / MARL
        ↓
    select useful signal action


LAYER 5
Safety Supervisor

    Is the action safe?

    Check:

        minimum green
        yellow interval
        clearance
        pedestrian constraints
        conflict constraints

    Unsafe action
        ↓
    REJECT

    Safe action
        ↓
    EXECUTE


LAYER 6
Monitoring / Digital Twin / Feedback

    Did the action produce the expected result?

    Observe
        ↓
    compare prediction vs reality
        ↓
    update state
        ↓
    continue control


============================================================
15. THE COMPLETE IDEA
============================================================

REAL WORLD
    |
    v
GPS + CCTV
    |
    v
OBSERVATIONS
    |
    v
LAYER 1
STATE ESTIMATION
    |
    +--> LWR
    |
    +--> Shockwaves
    |
    +--> Queue reconstruction
    |
    v
CURRENT TRAFFIC STATE
    |
    v
LAYER 2
FUTURE TRAFFIC PREDICTION
    |
    v
LAYER 3
NETWORK / MULTI-JUNCTION MODEL
    |
    v
LAYER 4
PPO / MARL
    |
    v
LAYER 5
SAFETY SUPERVISOR
    |
    v
TRAFFIC SIGNAL
    |
    v
NEW TRAFFIC STATE
    |
    +----------------------+
                           |
                           v
                    BACK TO LAYER 1


============================================================
KEY PRINCIPLE
============================================================

Physics
    !=
AI
    !=
Control

LWR and shockwave theory describe traffic physics.

State estimation reconstructs what cannot be directly observed.

Machine learning learns patterns from incomplete observations.

Prediction estimates what will happen next.

MARL/PPO chooses control actions.

The safety layer decides whether those actions are allowed.

This separation is fundamental to ASTRID.