import traci
import json
from sensor_simulator import (
    get_sensor_data,
    SIMULATION_END
)


SUMO_BINARY = "sumo-gui"

traci.start([
    SUMO_BINARY,
    "-c", "../sq.sumo.cfg"
])


TLS_ID = "0"




# ============================================================
# TRAFFIC LIGHT: CONTROLLED LINKS
# ============================================================

links = traci.trafficlight.getControlledLinks(TLS_ID)

for i, link in enumerate(links):
    print(i, "->", link)
    
 
INCOMING_EDGES = { 
    "1i", 
    "2i",
    "3i", 
    "4i" }

# ============================================================
# INCOMING EDGE → DIRECTION
# ============================================================

DIRECTION_EDGES = {
    "north": "4i",
    "south": "3i",
    "east": "2i",
    "west": "1i"
}


# A movement function 

def get_movement(route, route_index):
    
    """
        Logic Breakdown:
        Suppose SUMO gives:
            route = ['1i', '4o', '54o']
            route_index = 0

        Then:
            current_edge = '1i'
            next_edge    = '4o'
            movement     = '1i_to_4o'

        But if:
            route_index = 1

        Then:
            current_edge = '4o'
            Because '4o' is not an incoming edge:
            movement     = 'unknown'

            That is correct because the vehicle has already passed through the junction.
        """

    if not route:
        return "unknown"

    current_edge = route[route_index]

    # Vehicle is not currently on an incoming junction edge
    if current_edge not in INCOMING_EDGES:
        return "unknown"

    # There is no next edge
    if route_index + 1 >= len(route):
        return "unknown"

    next_edge = route[route_index + 1]

    # Movement through the junction
    return f"{current_edge}_to_{next_edge}"


# ============================================================
# INSPECT INCOMING LANES
# ============================================================

for direction, edge_id in DIRECTION_EDGES.items():

    lanes = traci.edge.getLaneNumber(edge_id)

    print("\n==============================")
    print(
        f"{direction.upper()} | EDGE: {edge_id}"
    )
    print("==============================")

    for lane_index in range(lanes):

        lane_id = f"{edge_id}_{lane_index}"

        length = traci.lane.getLength(lane_id)
        shape = traci.lane.getShape(lane_id)

        print(
            f"{lane_id}: "
            f"length={length:.2f} "
            f"start={shape[0]} "
            f"end={shape[-1]}"
        )


# ============================================================
# INITIALIZE FLOW TRACKING
# ============================================================

previous_edge_vehicles = {
    edge_id: set()
    for edge_id in DIRECTION_EDGES.values()
}

# ============================================================
# DATASET
# ============================================================

dataset = []


# ============================================================
# PREVIOUS VEHICLES
# ============================================================
#
# Used to calculate directional flow.
#
# At the beginning there are no previously observed vehicles.
# ============================================================

previous_edge_vehicles = {
    edge_id: set()
    for edge_id in DIRECTION_EDGES.values()
}


# ============================================================
# SIMULATION LOOP
# ============================================================

print("Simulation end:", SIMULATION_END)

for step in range(SIMULATION_END):

    # --------------------------------------------------------
    # Advance SUMO by one timestep
    # --------------------------------------------------------

    traci.simulationStep()
    
    # 1. Ground truth
    # 2. Traffic state
    # 3. GPS
    # 4. CCTV
    # 5. Camera aggregation
    # 6. Create ONE complete record
    # 7. Append record
    # 8. Display


    # ========================================================
    # 1. GROUND TRUTH VEHICLES
    # ========================================================

    vehicle_ids = traci.vehicle.getIDList()


    # --------------------------------------------------------
    # Group vehicles by incoming edge
    # --------------------------------------------------------

    edge_vehicles = {
        edge_id: []
        for edge_id in DIRECTION_EDGES.values()
    }


    for vehicle_id in vehicle_ids:

        edge_id = traci.vehicle.getRoadID(vehicle_id)

        if edge_id in edge_vehicles:

            edge_vehicles[edge_id].append(vehicle_id)


    # ========================================================
    # 2. DIRECTIONAL STATE
    # ========================================================

    state = {}


    for direction, edge_id in DIRECTION_EDGES.items():

        vehicles_on_edge = edge_vehicles[edge_id]


        # ----------------------------------------------------
        # Vehicle count
        # ----------------------------------------------------

        vehicle_count = len(vehicles_on_edge)


        # ----------------------------------------------------
        # Average speed
        # ----------------------------------------------------

        if vehicles_on_edge:

            speeds = [
                traci.vehicle.getSpeed(vehicle_id)
                for vehicle_id in vehicles_on_edge
            ]

            average_speed = sum(speeds) / len(speeds)

        else:

            average_speed = 0.0


        # ----------------------------------------------------
        # Queue
        # ----------------------------------------------------

        queue = sum(
            1
            for vehicle_id in vehicles_on_edge
            if traci.vehicle.getSpeed(vehicle_id) < 0.5
        )


        # ----------------------------------------------------
        # Flow
        #
        # Number of vehicles that appeared on this edge
        # since the previous timestep.
        # ----------------------------------------------------

        current_vehicle_set = set(vehicles_on_edge)

        new_vehicles = (
            current_vehicle_set
            - previous_edge_vehicles[edge_id]
        )

        flow = len(new_vehicles)


        # ----------------------------------------------------
        # Save directional state
        # ----------------------------------------------------

        state[direction] = {

            "vehicles": vehicle_count,

            "queue": queue,

            "speed": round(
                average_speed,
                2
            ),

            "flow": flow
        }


        # ----------------------------------------------------
        # Update previous vehicles
        # ----------------------------------------------------

        previous_edge_vehicles[edge_id] = (
            current_vehicle_set
        )


    # ========================================================
    # 3. SIMULATED SENSORS
    # ========================================================

    sensor_data = get_sensor_data()


    gps_count = len(
        sensor_data["gps"]
    )


    cctv_observations = sensor_data["cctv"]


    cctv_count = len(
        cctv_observations
    )


    # ========================================================
    # 4. CAMERA-BY-CAMERA AGGREGATION
    # ========================================================

    camera_counts = {

        "north_camera": 0,

        "south_camera": 0,

        "east_camera": 0,

        "west_camera": 0
    }


    for detection in cctv_observations:

        camera_id = detection["camera_id"]

        if camera_id in camera_counts:

            camera_counts[camera_id] += 1


    # ========================================================
    # 5. CREATE ONE COMPLETE DATASET RECORD
    # ========================================================

    record = {

        "step": step,

        # -------------------------------
        # Ground truth
        # -------------------------------

        "traffic": state,

        # -------------------------------
        # Sensor measurements
        # -------------------------------

        "gps_count": gps_count,

        "cctv_count": cctv_count,

        # -------------------------------
        # Camera measurements
        # -------------------------------

        "camera_counts": camera_counts,

        # -------------------------------
        # Raw simulated sensor data
        # -------------------------------

        "sensors": sensor_data
    }


    # ========================================================
    # 6. APPEND TO DATASET
    # ========================================================

    dataset.append(record)


    # ========================================================
    # 7. DISPLAY
    # ========================================================

    print(
        f"\nStep: {step:3d}"
    )


    for direction in [
        "north",
        "south",
        "east",
        "west"
    ]:

        direction_state = state[direction]

        print(
            f"{direction.upper():5s} | "
            f"Vehicles: {direction_state['vehicles']:3d} | "
            f"Queue: {direction_state['queue']:3d} | "
            f"Speed: {direction_state['speed']:5.2f} m/s | "
            f"Flow: {direction_state['flow']:2d}"
        )


    print(
        f"CCTV Total: {cctv_count:3d} | "
        f"GPS: {gps_count:3d}"
    )


    print(
        f"Cameras: "
        f"N={camera_counts['north_camera']:3d} | "
        f"S={camera_counts['south_camera']:3d} | "
        f"E={camera_counts['east_camera']:3d} | "
        f"W={camera_counts['west_camera']:3d}"
    )


# ============================================================
# SAVE DATASET
# ============================================================

with open("sensor_dataset.json", "w") as f:

    json.dump(
        dataset,
        f,
        indent=2
    )


print(
    f"\nDataset saved: {len(dataset)} records"
)


# ============================================================
# CLOSE SUMO
# ============================================================

traci.close()