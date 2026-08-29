"""
Run/read the SUMO simulation state and produce synthetic
sensor observations.

For example:

Step 0
├── Ground truth vehicles
├── GPS observations
├── CCTV observations
├── Traffic state
└── Signal state

Step 1
├── Ground truth vehicles
├── GPS observations
├── CCTV observations
├── Traffic state
└── Signal state

This file does NOT create a database.

Architecture
------------

SUMO
 ↓
TraCI
 ↓
sensor_simulator.py
 ↓
GPS + CCTV + ground truth
 ↓
state_extractor.py
 ↓
sensor_dataset.json


Scenario architecture
---------------------

scenario_001/scenario.json
        ↓
SensorSimulator
        ↓
SUMO observations

scenario_002/scenario.json
        ↓
SensorSimulator
        ↓
SUMO observations

...

scenario_200/scenario.json
        ↓
SensorSimulator
        ↓
SUMO observations


IMPORTANT
---------

There is only ONE copy of sensor_simulator.py.

The same code is reused for every scenario.

The scenario-specific values come from scenario.json:

    gps_penetration
    cctv_detection
    seed
    simulation_end

This module does NOT create vehicles.

SUMO creates the traffic population.

This module only observes those vehicles and applies
the sensor configuration belonging to the current scenario.

Incoming roads are approximately 484.9 m long.

Each CCTV camera observes only the final 150 m
approaching the junction.
"""


from dataclasses import dataclass
from typing import Set, List
import hashlib

import traci


# ============================================================
# CONSTANTS
# ============================================================

QUEUE_SPEED_THRESHOLD = 0.5

CAMERA_LENGTH = 150.0

INCOMING_EDGES = {
    "1i",
    "2i",
    "3i",
    "4i",
}

CAMERA_APPROACHES = {

    "north_camera": "4i",

    "south_camera": "3i",

    "east_camera": "2i",

    "west_camera": "1i",
}


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SensorConfig:
    """
    Sensor configuration for ONE scenario.

    These values come directly from scenario.json.
    """

    scenario_name: str

    seed: int

    gps_penetration: float

    cctv_detection: float

    simulation_end: int


# ============================================================
# SENSOR SIMULATOR
# ============================================================

class SensorSimulator:
    """
    Observe vehicles currently existing in SUMO.

    This class does not create or control vehicles.

    SUMO is the source of ground-truth traffic.

    SensorSimulator creates:

        ground_truth
        GPS observations
        CCTV observations
    """

    def __init__(
        self,
        config: SensorConfig,
    ):

        self.config = config

        # ----------------------------------------------------
        # Vehicles permanently selected as GPS probes.
        #
        # Once a vehicle is selected, it remains a GPS probe
        # for the rest of its lifetime.
        # ----------------------------------------------------

        self.gps_probe_vehicles: Set[str] = set()

        self._validate_config()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_config(self):

        if not (
            0.0
            <= self.config.gps_penetration
            <= 1.0
        ):

            raise ValueError(
                "gps_penetration must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.config.cctv_detection
            <= 1.0
        ):

            raise ValueError(
                "cctv_detection must be "
                "between 0 and 1."
            )

        if self.config.simulation_end <= 0:

            raise ValueError(
                "simulation_end must be > 0."
            )

        if self.config.seed < 0:

            raise ValueError(
                "seed must be >= 0."
            )

        if not self.config.scenario_name:

            raise ValueError(
                "scenario_name cannot be empty."
            )

    # ========================================================
    # DETERMINISTIC RANDOM VALUE
    # ========================================================

    def _random_value(
        self,
        vehicle_id: str,
        sensor_name: str,
    ) -> float:
        """
        Produce a deterministic pseudo-random value.

        The value depends on:

            scenario seed
            scenario name
            sensor name
            vehicle ID

        Therefore the same scenario produces reproducible
        sensor assignments.
        """

        key = (
            f"{self.config.seed}:"
            f"{self.config.scenario_name}:"
            f"{sensor_name}:"
            f"{vehicle_id}"
        )

        digest = hashlib.sha256(
            key.encode("utf-8")
        ).digest()

        integer = int.from_bytes(
            digest[:8],
            byteorder="big",
        )

        return integer / float(2**64)

    # ========================================================
    # GPS ASSIGNMENT
    # ========================================================

    def assign_gps_probe(
        self,
        vehicle_id: str,
    ) -> bool:
        """
        Determine whether this vehicle is a GPS probe.

        GPS penetration is applied once per vehicle.

        Example:

            gps_penetration = 0.30

        approximately 30% of vehicles will become
        persistent GPS probes.
        """

        if vehicle_id in self.gps_probe_vehicles:

            return True

        value = self._random_value(
            vehicle_id,
            "gps",
        )

        if (
            value
            <
            self.config.gps_penetration
        ):

            self.gps_probe_vehicles.add(
                vehicle_id
            )

            return True

        return False

    # ========================================================
    # MOVEMENT
    # ========================================================

    @staticmethod
    def get_movement(
        route: List[str],
        route_index: int,
    ) -> str:
        """
        Determine vehicle movement from its SUMO route.

        Example:

            4i -> 3o

        means:

            north -> south

        which is a straight movement.
        """

        if not route:

            return "unknown"

        if route_index < 0:

            return "unknown"

        if route_index >= len(route):

            return "unknown"

        current_edge = route[
            route_index
        ]

        if current_edge not in INCOMING_EDGES:

            return "unknown"

        if (
            route_index + 1
            >= len(route)
        ):

            return "unknown"

        next_edge = route[
            route_index + 1
        ]

        movement_map = {

            # North
            ("4i", "3o"):
                "north_to_south",

            ("4i", "2o"):
                "north_to_east",

            ("4i", "1o"):
                "north_to_west",

            # South
            ("3i", "4o"):
                "south_to_north",

            ("3i", "2o"):
                "south_to_east",

            ("3i", "1o"):
                "south_to_west",

            # East
            ("2i", "4o"):
                "east_to_north",

            ("2i", "3o"):
                "east_to_south",

            ("2i", "1o"):
                "east_to_west",

            # West
            ("1i", "4o"):
                "west_to_north",

            ("1i", "3o"):
                "west_to_south",

            ("1i", "2o"):
                "west_to_east",
        }

        return movement_map.get(
            (
                current_edge,
                next_edge,
            ),
            "unknown",
        )

    # ========================================================
    # CAMERA LOCATION
    # ========================================================

    @staticmethod
    def vehicle_in_camera(
        vehicle_id: str,
        camera_edge: str,
    ) -> bool:
        """
        Return True when a vehicle is inside the final
        CAMERA_LENGTH metres of an incoming road.

        Incoming road length is approximately 484.9 m.

        Therefore, with CAMERA_LENGTH = 150 m:

            camera region ≈ 334.9 m -> 484.9 m

        measured along the lane toward the junction.
        """

        edge_id = traci.vehicle.getRoadID(
            vehicle_id
        )

        if edge_id != camera_edge:

            return False

        lane_id = traci.vehicle.getLaneID(
            vehicle_id
        )

        if not lane_id:

            return False

        lane_length = traci.lane.getLength(
            lane_id
        )

        position = traci.vehicle.getLanePosition(
            vehicle_id
        )

        return (
            position
            >=
            lane_length - CAMERA_LENGTH
        )

    # ========================================================
    # GPS OBSERVATIONS
    # ========================================================

    def get_gps_observations(
        self,
        vehicle_ids,
        timestamp,
    ):

        observations = []

        for vehicle_id in vehicle_ids:

            if not self.assign_gps_probe(
                vehicle_id
            ):

                continue

            x, y = traci.vehicle.getPosition(
                vehicle_id
            )

            observations.append({

                "id":
                    vehicle_id,

                "position": {

                    "x":
                        round(x, 2),

                    "y":
                        round(y, 2),
                },

                "speed":
                    round(
                        traci.vehicle.getSpeed(
                            vehicle_id
                        ),
                        2,
                    ),

                "edge":
                    traci.vehicle.getRoadID(
                        vehicle_id
                    ),

                "lane":
                    traci.vehicle.getLaneID(
                        vehicle_id
                    ),

                "vehicle_type":
                    traci.vehicle.getTypeID(
                        vehicle_id
                    ),

                "timestamp":
                    timestamp,
            })

        return observations

    # ========================================================
    # CCTV OBSERVATIONS
    # ========================================================

    def get_cctv_observations(
        self,
        vehicle_ids,
        timestamp,
    ):

        observations = []

        for (
            camera_id,
            camera_edge
        ) in CAMERA_APPROACHES.items():

            for vehicle_id in vehicle_ids:

                if not self.vehicle_in_camera(
                    vehicle_id,
                    camera_edge,
                ):

                    continue

                detection_value = (
                    self._random_value(
                        vehicle_id,
                        camera_id,
                    )
                )

                if (
                    detection_value
                    >=
                    self.config.cctv_detection
                ):

                    continue

                x, y = traci.vehicle.getPosition(
                    vehicle_id
                )

                observations.append({

                    "camera_id":
                        camera_id,

                    "id":
                        vehicle_id,

                    "position": {

                        "x":
                            round(x, 2),

                        "y":
                            round(y, 2),
                    },

                    "speed":
                        round(
                            traci.vehicle.getSpeed(
                                vehicle_id
                            ),
                            2,
                        ),

                    "edge":
                        traci.vehicle.getRoadID(
                            vehicle_id
                        ),

                    "lane":
                        traci.vehicle.getLaneID(
                            vehicle_id
                        ),

                    "lane_position":
                        round(
                            traci.vehicle.getLanePosition(
                                vehicle_id
                            ),
                            2,
                        ),

                    "vehicle_type":
                        traci.vehicle.getTypeID(
                            vehicle_id
                        ),

                    "timestamp":
                        timestamp,
                })

        return observations

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    def get_ground_truth(
        self,
        vehicle_ids,
        timestamp,
    ):

        ground_truth = []

        for vehicle_id in vehicle_ids:

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

            lane_position = (
                traci.vehicle.getLanePosition(
                    vehicle_id
                )
            )

            route = list(
                traci.vehicle.getRoute(
                    vehicle_id
                )
            )

            route_index = (
                traci.vehicle.getRouteIndex(
                    vehicle_id
                )
            )

            movement = self.get_movement(
                route,
                route_index,
            )

            ground_truth.append({

                "id":
                    vehicle_id,

                "position": {

                    "x":
                        round(x, 2),

                    "y":
                        round(y, 2),
                },

                "speed":
                    round(
                        speed,
                        2,
                    ),

                "edge":
                    edge,

                "lane":
                    lane,

                "lane_position":
                    round(
                        lane_position,
                        2,
                    ),

                "route":
                    route,

                "route_index":
                    route_index,

                "movement":
                    movement,

                "vehicle_type":
                    traci.vehicle.getTypeID(
                        vehicle_id
                    ),

                "timestamp":
                    timestamp,
            })

        return ground_truth

    # ========================================================
    # MAIN INTERFACE
    # ========================================================

    def get_sensor_data(self):
        """
        Collect all sensor observations for the current
        SUMO timestep.

        SUMO must already have been advanced with:

            traci.simulationStep()

        before this function is called.
        """

        timestamp = traci.simulation.getTime()

        vehicle_ids = (
            traci.vehicle.getIDList()
        )

        return {

            "timestamp":
                timestamp,

            "ground_truth":
                self.get_ground_truth(
                    vehicle_ids,
                    timestamp,
                ),

            "gps":
                self.get_gps_observations(
                    vehicle_ids,
                    timestamp,
                ),

            "cctv":
                self.get_cctv_observations(
                    vehicle_ids,
                    timestamp,
                ),
        }