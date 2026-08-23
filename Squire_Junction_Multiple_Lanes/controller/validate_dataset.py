import json
from statistics import mean, stdev


DATASET_FILE = "sensor_dataset.json"


# ============================================================
# LOAD DATASET
# ============================================================

with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)


print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n[1] BASIC INFORMATION")

print("Records:", len(dataset))

if not dataset:
    raise ValueError("Dataset is empty.")


print("First step:", dataset[0]["step"])
print("Last step :", dataset[-1]["step"])


# ============================================================
# CHECK STEP SEQUENCE
# ============================================================

print("\n[2] STEP SEQUENCE")

steps = [record["step"] for record in dataset]

expected_steps = list(range(len(dataset)))

if steps == expected_steps:
    print("PASS: Steps are continuous.")
else:
    print("WARNING: Step sequence is not continuous.")
    print("Actual:", steps[:20])


# ============================================================
# CHECK REQUIRED FIELDS
# ============================================================

print("\n[3] RECORD STRUCTURE")

required_record_keys = [
    "step",
    "traffic",
    "gps_count",
    "cctv_count",
    "camera_counts",
    "sensors",
]

for key in required_record_keys:

    if key in dataset[0]:
        print(f"PASS: {key}")
    else:
        print(f"FAIL: Missing {key}")


# ============================================================
# TRAFFIC DIRECTIONS
# ============================================================

directions = [
    "north",
    "south",
    "east",
    "west",
]


traffic_fields = [
    "vehicles",
    "queue",
    "speed",
    "flow",
]


print("\n[4] TRAFFIC STRUCTURE")

for direction in directions:

    print(f"\n{direction.upper()}")

    for field in traffic_fields:

        if field in dataset[0]["traffic"][direction]:
            print(f"  PASS: {field}")
        else:
            print(f"  FAIL: missing {field}")


# ============================================================
# STATISTICS
# ============================================================

print("\n[5] TRAFFIC STATISTICS")


for direction in directions:

    print(f"\n{direction.upper()}")

    for field in traffic_fields:

        values = [
            record["traffic"][direction][field]
            for record in dataset
        ]

        minimum = min(values)
        maximum = max(values)
        average = mean(values)

        if len(values) > 1:
            variation = stdev(values)
        else:
            variation = 0.0

        print(
            f"{field:8s} "
            f"min={minimum:8.2f} "
            f"max={maximum:8.2f} "
            f"mean={average:8.2f} "
            f"std={variation:8.2f}"
        )


# ============================================================
# SENSOR STATISTICS
# ============================================================

print("\n[6] SENSOR STATISTICS")


gps_counts = [
    record["gps_count"]
    for record in dataset
]

cctv_counts = [
    record["cctv_count"]
    for record in dataset
]


print(
    "GPS :",
    f"min={min(gps_counts)},",
    f"max={max(gps_counts)},",
    f"mean={mean(gps_counts):.2f}"
)


print(
    "CCTV:",
    f"min={min(cctv_counts)},",
    f"max={max(cctv_counts)},",
    f"mean={mean(cctv_counts):.2f}"
)


# ============================================================
# CAMERA STATISTICS
# ============================================================

print("\n[7] CAMERA STATISTICS")


camera_names = [
    "north_camera",
    "south_camera",
    "east_camera",
    "west_camera",
]


for camera in camera_names:

    values = [
        record["camera_counts"][camera]
        for record in dataset
    ]

    print(
        f"{camera:15s}"
        f" min={min(values):3d}"
        f" max={max(values):3d}"
        f" mean={mean(values):6.2f}"
    )


# ============================================================
# GROUND TRUTH STATISTICS
# ============================================================

print("\n[8] GROUND TRUTH")

vehicle_counts = [
    len(record["sensors"]["ground_truth"])
    for record in dataset
]


print(
    "Vehicles per timestep:",
    f"min={min(vehicle_counts)},",
    f"max={max(vehicle_counts)},",
    f"mean={mean(vehicle_counts):.2f}"
)


# ============================================================
# VEHICLE TYPES
# ============================================================

print("\n[9] VEHICLE TYPE DISTRIBUTION")


vehicle_type_counts = {}


for record in dataset:

    for vehicle in record["sensors"]["ground_truth"]:

        vehicle_type = vehicle["vehicle_type"]

        vehicle_type_counts[vehicle_type] = (
            vehicle_type_counts.get(vehicle_type, 0) + 1
        )


total_vehicle_observations = sum(
    vehicle_type_counts.values()
)


for vehicle_type, count in sorted(
    vehicle_type_counts.items()
):

    percentage = (
        100 * count / total_vehicle_observations
    )

    print(
        f"{vehicle_type:8s}"
        f" count={count:5d}"
        f" percentage={percentage:6.2f}%"
    )


# ============================================================
# MOVEMENT DISTRIBUTION
# ============================================================

print("\n[10] MOVEMENT DISTRIBUTION")


movement_counts = {}


for record in dataset:

    for vehicle in record["sensors"]["ground_truth"]:

        movement = vehicle["movement"]

        movement_counts[movement] = (
            movement_counts.get(movement, 0) + 1
        )


for movement, count in sorted(
    movement_counts.items()
):

    print(
        f"{movement:20s}: {count}"
    )


# ============================================================
# SENSOR COVERAGE
# ============================================================

print("\n[11] SENSOR COVERAGE")


gps_zero_steps = sum(
    1
    for value in gps_counts
    if value == 0
)


cctv_zero_steps = sum(
    1
    for value in cctv_counts
    if value == 0
)


print(
    "GPS zero-observation steps:",
    gps_zero_steps,
    "/",
    len(dataset)
)


print(
    "CCTV zero-observation steps:",
    cctv_zero_steps,
    "/",
    len(dataset)
)


# ============================================================
# VARIATION CHECK
# ============================================================

print("\n[12] VARIATION CHECK")


for direction in directions:

    for field in [
        "vehicles",
        "queue",
        "speed",
        "flow",
    ]:

        values = [
            record["traffic"][direction][field]
            for record in dataset
        ]

        unique_values = len(set(values))

        if unique_values > 1:

            print(
                f"PASS: {direction}.{field}"
                f" has {unique_values} unique values"
            )

        else:

            print(
                f"WARNING: {direction}.{field}"
                " is constant"
            )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)