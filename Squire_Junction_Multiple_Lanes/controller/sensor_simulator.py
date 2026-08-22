import traci
import random



# ============================================================
# SENSOR CONFIGURATION: not add fake camera errors, GPS noise, missed detections, occlusion, etc. yet.
# ============================================================

GPS_PENETRATION = 0.20

QUEUE_SPEED_THRESHOLD = 0.5


# ============================================================
# PERSISTENT GPS PROBE ASSIGNMENT
# ============================================================

gps_probe_vehicles = set()
random_generator = random.Random(42)

def assign_gps_probe(vehicle_id):
    """
    Assign GPS status once when a vehicle first appears.

    The assignment remains fixed for the vehicle's
    lifetime in the simulation.
    """

    if vehicle_id in gps_probe_vehicles:
        return True

    if random_generator.random() < GPS_PENETRATION:
        gps_probe_vehicles.add(vehicle_id)
        return True

    return False
 

# ============================================================
# CCTV CAMERA CONFIGURATION
# ============================================================

CAMERA_APPROACHES = {
    "north_camera": "4i",
    "south_camera": "3i",
    "east_camera": "2i",
    "west_camera": "1i",
}

# How much of the incoming road the camera observes
CAMERA_LENGTH = 150.0


def vehicle_in_camera(vehicle_id, camera_edge):
    """
    Check whether a vehicle is inside the final
    CAMERA_LENGTH metres of an incoming approach.
    """

    edge_id = traci.vehicle.getRoadID(vehicle_id)

    if edge_id != camera_edge:
        return False

    lane_id = traci.vehicle.getLaneID(vehicle_id)

    lane_length = traci.lane.getLength(lane_id)

    position_on_lane = traci.vehicle.getLanePosition(vehicle_id)

    return position_on_lane >= lane_length - CAMERA_LENGTH


# ============================================================
# GPS OBSERVATIONS
# ============================================================

def get_gps_observations(vehicle_ids, timestamp):
    """
    Generate individual GPS/probe observations.

    Each vehicle either belongs to the persistent probe
    population or does not.
    """

    observations = []

    for vehicle_id in vehicle_ids:

        if not assign_gps_probe(vehicle_id):
            continue

        x, y = traci.vehicle.getPosition(vehicle_id)

        observations.append({
            "id": vehicle_id,

            "position": {
                "x": round(x, 2),
                "y": round(y, 2)
            },

            "speed": round(
                traci.vehicle.getSpeed(vehicle_id),
                2
            ),

            "edge": traci.vehicle.getRoadID(
                vehicle_id
            ),

            "lane": traci.vehicle.getLaneID(
                vehicle_id
            ),

            "vehicle_type": traci.vehicle.getTypeID(
                vehicle_id
            ),

            "timestamp": timestamp
        })

    return observations


# ============================================================
# CCTV OBSERVATIONS
# ============================================================

def get_cctv_observations(vehicle_ids, timestamp):

    observations = []

    for camera_id, camera_edge in CAMERA_APPROACHES.items():

        for vehicle_id in vehicle_ids:

            if not vehicle_in_camera(
                vehicle_id,
                camera_edge
            ):
                continue

            x, y = traci.vehicle.getPosition(vehicle_id)

            speed = traci.vehicle.getSpeed(vehicle_id)

            observations.append({
                "camera_id": camera_id,

                "id": vehicle_id,

                "position": {
                    "x": round(x, 2),
                    "y": round(y, 2)
                },

                "speed": round(speed, 2),

                "edge": traci.vehicle.getRoadID(
                    vehicle_id
                ),

                "lane": traci.vehicle.getLaneID(
                    vehicle_id
                ),

                "lane_position": round(
                    traci.vehicle.getLanePosition(
                        vehicle_id
                    ),
                    2
                ),

                "vehicle_type": traci.vehicle.getTypeID(
                    vehicle_id
                ),

                "timestamp": timestamp
            })

    return observations


# ============================================================
# MOVEMENT INFERENCE FOR GROUND TRUTH
# ============================================================

def get_movement(route):
    """
    Determine the vehicle's movement from its SUMO route.

    The route tells us the sequence of road edges.

        4i -> 3o

    The movement is the higher-level interpretation:

        north -> south

    IMPORTANT:
    This function is used only for SUMO ground truth.
    GPS and CCTV do NOT receive this movement directly.

    Later, the neural network will learn to estimate
    this hidden movement from GPS/CCTV observations.
    """

    if len(route) < 2:
        return "unknown"

    incoming = route[0]
    outgoing = route[1]

    movement_map = {

        # NORTH
        ("4i", "3o"): "north_to_south",
        ("4i", "2o"): "north_to_east",
        ("4i", "1o"): "north_to_west",

        # SOUTH
        ("3i", "4o"): "south_to_north",
        ("3i", "2o"): "south_to_east",
        ("3i", "1o"): "south_to_west",

        # EAST
        ("2i", "4o"): "east_to_north",
        ("2i", "3o"): "east_to_south",
        ("2i", "1o"): "east_to_west",

        # WEST
        ("1i", "4o"): "west_to_north",
        ("1i", "3o"): "west_to_south",
        ("1i", "2o"): "west_to_east",
    }

    return movement_map.get(
        (incoming, outgoing),
        "unknown"
    )


# ============================================================
# MAIN SENSOR INTERFACE
# ============================================================

def get_sensor_data():

    timestamp = traci.simulation.getTime()

    vehicle_ids = traci.vehicle.getIDList()


    # ========================================================
    # 1. GPS
    # ========================================================

    gps_observations = get_gps_observations(
        vehicle_ids,
        timestamp
    )


    # ========================================================
    # 2. CCTV
    # ========================================================

    cctv_observations = get_cctv_observations(
        vehicle_ids,
        timestamp
    )


    # ========================================================
    # 3. SUMO GROUND TRUTH
    # ========================================================

    ground_truth = []

    for vehicle_id in vehicle_ids:

        # --------------------------------------------
        # Current physical state
        # --------------------------------------------

        x, y = traci.vehicle.getPosition(
            vehicle_id
        )

        speed = traci.vehicle.getSpeed(
            vehicle_id
        )

        edge = traci.vehicle.getRoadID(
            vehicle_id
        )

        lane = traci.vehicle.getLaneID(
            vehicle_id
        )

        lane_position = traci.vehicle.getLanePosition(
            vehicle_id
        )


        # --------------------------------------------
        # SUMO route
        # --------------------------------------------

        route = list(
            traci.vehicle.getRoute(
                vehicle_id
            )
        )


        # --------------------------------------------
        # Route index
        # --------------------------------------------

        route_index = traci.vehicle.getRouteIndex(
            vehicle_id
        )


        # --------------------------------------------
        # Higher-level movement
        # --------------------------------------------

        movement = get_movement(
            route
        )


        # --------------------------------------------
        # Store ground truth
        # --------------------------------------------

        ground_truth.append({

            "id": vehicle_id,

            "position": {
                "x": round(x, 2),
                "y": round(y, 2)
            },

            "speed": round(
                speed,
                2
            ),

            "edge": edge,

            "lane": lane,

            "lane_position": round(
                lane_position,
                2
            ),

            "route": route,

            "route_index": route_index,

            "movement": movement,

            "vehicle_type": traci.vehicle.getTypeID(
                vehicle_id
            ),

            "timestamp": timestamp
        })

  
    # ========================================================
    # 4. RETURN ALL DATA
    # ========================================================

    return {

        "timestamp": timestamp,

        "ground_truth": ground_truth,

        "gps": gps_observations,

        "cctv": cctv_observations
    }
    


    
    