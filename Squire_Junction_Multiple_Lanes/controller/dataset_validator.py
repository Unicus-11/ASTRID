import json


DATASET_FILE = "sensor_dataset.json"


# ============================================================
# LOAD DATASET
# ============================================================

with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)


print("========================================")
print("ASTRID DATASET VALIDATION")
print("========================================")


# ============================================================
# BASIC INFORMATION
# ============================================================

print(f"Timesteps: {len(dataset)}")

if dataset:
    print(
        f"First timestamp: {dataset[0]['timestamp']}"
    )

    print(
        f"Last timestamp: {dataset[-1]['timestamp']}"
    )


# ============================================================
# REQUIRED FIELDS
# ============================================================

required_timestep_fields = [
    "timestamp",
    "ground_truth",
    "gps",
    "cctv"
]

required_vehicle_fields = [
    "id",
    "position",
    "speed",
    "edge",
    "lane",
    "lane_position",
    "route",
    "route_index",
    "movement",
    "vehicle_type",
    "timestamp"
]


missing_fields = 0
unknown_movements = 0
invalid_speeds = 0
invalid_positions = 0


# ============================================================
# VALIDATE
# ============================================================

for timestep in dataset:

    # Check timestep fields
    for field in required_timestep_fields:

        if field not in timestep:
            missing_fields += 1

    # Check ground truth
    for vehicle in timestep.get("ground_truth", []):

        for field in required_vehicle_fields:

            if field not in vehicle:
                missing_fields += 1

        # Check movement
        if vehicle.get("movement") == "unknown":
            unknown_movements += 1

        # Check speed
        if vehicle.get("speed", -1) < 0:
            invalid_speeds += 1

        # Check position
        position = vehicle.get("position", {})

        if "x" not in position or "y" not in position:
            invalid_positions += 1


# ============================================================
# RESULTS
# ============================================================

print()
print("Missing required fields:", missing_fields)
print("Unknown movements:", unknown_movements)
print("Invalid speeds:", invalid_speeds)
print("Invalid positions:", invalid_positions)

print()
print("========================================")

if (
    missing_fields == 0
    and unknown_movements == 0
    and invalid_speeds == 0
    and invalid_positions == 0
):

    print("RESULT: DATASET PASSED")

else:

    print("RESULT: DATASET HAS WARNINGS")

print("========================================")