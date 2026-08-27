"""
pipeline is:

baseline.json
      ↓
generate_flows.py
      ↓
sq.flow.xml
      ↓
SUMO
      ↓
normal_controller
      ↓
state_extractor
      ↓
sensor_simulator
      ↓
sensor_dataset.json

One important condition: this is true only after we connect normal_controller.py to the same SUMO run in state_extractor.py. Then sensor_dataset.json will be the normal-controller dataset. """

import traci


TLS_ID = "0"


# ============================================================
# NORMAL WEBSTER BASELINE
# ============================================================

def apply_normal_controller(scenario):

    # Demand is expressed in vehicles/hour.
    demand_rate = {
        "low": 1000,
        "medium": 1600,
        "high": 2000,
        "very_high": 2500,
    }

    demand = demand_rate[
        scenario["demand"]
    ]

    approach = scenario[
        "approach_distribution"
    ]

    # Convert total demand into directional demand.
    NS = demand * (
        approach["north"]
        + approach["south"]
    )

    EW = demand * (
        approach["east"]
        + approach["west"]
    )

    W_NS = 6
    W_EW = 6

    S_NS = 525 * W_NS
    S_EW = 525 * W_EW

    y_NS = NS / S_NS
    y_EW = EW / S_EW

    Y = y_NS + y_EW

    L = 14

    if Y < 1:
        C = (1.5 * L + 5) / (1 - Y)
        C = max(60, min(C, 120))
    else:
        C = 120

    green_total = C - L

    # Avoid division by zero.
    if Y > 0:
        g_NS = green_total * (y_NS / Y)
        g_EW = green_total * (y_EW / Y)
    else:
        g_NS = green_total / 2
        g_EW = green_total / 2

    print("\n=== NORMAL WEBSTER SIGNAL ===")

    print("Demand :", demand, "veh/h")
    print("NS     :", round(NS, 1), "veh/h")
    print("EW     :", round(EW, 1), "veh/h")
    print("Y      :", round(Y, 3))
    print("Cycle  :", round(C, 1), "seconds")
    print("NS Green:", round(g_NS, 1), "seconds")
    print("EW Green:", round(g_EW, 1), "seconds")

    program = (
        traci.trafficlight
        .getAllProgramLogics(TLS_ID)[0]
    )

    phases = program.getPhases()

    phases[0].duration = g_EW
    phases[1].duration = 3

    phases[2].duration = g_EW
    phases[3].duration = 3

    phases[4].duration = g_NS
    phases[5].duration = 3

    phases[6].duration = g_NS
    phases[7].duration = 3

    program.phases = phases

    traci.trafficlight.setProgramLogic(
        TLS_ID,
        program,
    )

    print("Signal timing applied.")