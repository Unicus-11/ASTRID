import traci
import json
from sensor_simulator import get_sensor_data


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


# ============================================================
# INCOMING EDGE → DIRECTION
# ============================================================

DIRECTION_EDGES = {
    "north": "4i",
    "south": "3i",
    "east": "2i",
    "west": "1i"
}


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
# SIMULATION LOOP
# ============================================================

for step in range(100):

    traci.simulationStep()


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
        # Speed
        # ----------------------------------------------------

        if vehicles_on_edge:

            speeds = [
                traci.vehicle.getSpeed(vehicle_id)
                for vehicle_id in vehicles_on_edge
            ]

            average_speed = (
                sum(speeds) / len(speeds)
            )

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
        # Directional flow
        #
        # Vehicles that were NOT on this edge in the
        # previous timestep but are on it now.
        # ----------------------------------------------------

        current_vehicle_set = set(
            vehicles_on_edge
        )

        new_vehicles = (
            current_vehicle_set
            - previous_edge_vehicles[edge_id]
        )

        flow = len(new_vehicles)


        # ----------------------------------------------------
        # Save state
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


        # Update previous vehicles
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
    #
    # camera_id is intentionally preserved.
    #
    # This gives:
    #
    # north_camera → detections
    # south_camera → detections
    # east_camera  → detections
    # west_camera  → detections
    #
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



# ============================================================
# DATASET
# ============================================================

dataset = []


# ============================================================
# SIMULATION LOOP
# ============================================================

for step in range(100):

    traci.simulationStep()

    sensor_data = get_sensor_data()

    dataset.append(sensor_data)


    # ========================================================
    # 5. DISPLAY
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



# ============================================================
# CLOSE SUMO
# ============================================================

traci.close()