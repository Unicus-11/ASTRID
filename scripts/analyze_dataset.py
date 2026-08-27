import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCENARIO_PATH = (
    PROJECT_ROOT
    / "scenarios"
    / "baseline"
    / "scenario.json"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensor_dataset.json"
)


# ============================================================
# LOAD SCENARIO
# ============================================================

with open(
    SCENARIO_PATH,
    "r",
    encoding="utf-8"
) as f:

    SCENARIO = json.load(f)


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DATASET ANALYSIS")
print("=" * 70)

print()
print("Scenario:", SCENARIO["name"])
print("Expected vehicles:", SCENARIO["total_vehicles"])
print("GPS penetration:", SCENARIO["gps_penetration"])
print("CCTV detection:", SCENARIO["cctv_detection"])


# ============================================================
# BASIC
# ============================================================

print("\n[1] BASIC")

print(
    "Records:",
    len(dataset)
)


if dataset:

    print(
        "Steps:",
        dataset[0]["step"],
        "->",
        dataset[-1]["step"]
    )


# ============================================================
# VEHICLE OBSERVATIONS
# ============================================================

all_vehicle_observations = []

for record in dataset:

    vehicles = record[
        "sensors"
    ][
        "ground_truth"
    ]

    all_vehicle_observations.extend(
        vehicles
    )


print("\n[2] VEHICLE OBSERVATIONS")

print(
    "Total observations:",
    len(all_vehicle_observations)
)


unique_vehicle_ids = set(

    vehicle["id"]

    for vehicle
    in all_vehicle_observations
)


print(
    "Unique vehicle IDs:",
    len(unique_vehicle_ids)
)


# ============================================================
# VEHICLE TYPE
# ============================================================

print("\n[3] VEHICLE TYPES")

type_counter = Counter(

    vehicle["vehicle_type"]

    for vehicle
    in all_vehicle_observations
)


total = sum(
    type_counter.values()
)


for vehicle_type, count in sorted(
    type_counter.items()
):

    percentage = (
        100 * count / total
    )

    print(

        f"{vehicle_type:6s} "
        f"{count:8d} "
        f"{percentage:6.2f}%"
    )


# ============================================================
# MOVEMENT
# ============================================================

print("\n[4] MOVEMENTS")

movement_counter = Counter(

    vehicle["movement"]

    for vehicle
    in all_vehicle_observations
)


for movement, count in sorted(
    movement_counter.items()
):

    print(

        f"{movement:20s} : "
        f"{count:8d}"
    )


# ============================================================
# SPEED
# ============================================================

print("\n[5] SPEED")

speeds = [

    vehicle["speed"]

    for vehicle
    in all_vehicle_observations
]


print(
    "min :",
    round(min(speeds), 2)
)

print(
    "max :",
    round(max(speeds), 2)
)

print(
    "mean:",
    round(
        statistics.mean(speeds),
        2
    )
)


if len(speeds) > 1:

    print(
        "std :",
        round(
            statistics.stdev(speeds),
            2
        )
    )


# ============================================================
# STOPPED VEHICLES
# ============================================================

print("\n[6] STOPPED VEHICLES")

stopped = sum(

    1

    for speed in speeds

    if speed < 0.5
)


print(
    "speed < 0.5 m/s:",
    stopped
)


print(
    "percentage:",
    round(
        100 * stopped / len(speeds),
        2
    ),
    "%"
)


# ============================================================
# TEMPORAL VEHICLE TRACKING
# ============================================================

print("\n[7] TEMPORAL VEHICLE TRACKING")

vehicle_history = defaultdict(list)


for record in dataset:

    for vehicle in record[
        "sensors"
    ][
        "ground_truth"
    ]:

        vehicle_history[
            vehicle["id"]
        ].append(

            (
                vehicle["timestamp"],
                vehicle["speed"]
            )
        )


trajectory_lengths = [

    len(history)

    for history
    in vehicle_history.values()
]


print(
    "Vehicles observed:",
    len(trajectory_lengths)
)


print(
    "Minimum observations/vehicle:",
    min(trajectory_lengths)
)


print(
    "Maximum observations/vehicle:",
    max(trajectory_lengths)
)


print(
    "Mean observations/vehicle:",
    round(
        statistics.mean(
            trajectory_lengths
        ),
        2
    )
)


# ============================================================
# TRAFFIC VARIABLES
# ============================================================

print("\n[8] TRAFFIC VARIABLES")


for direction in [
    "north",
    "south",
    "east",
    "west"
]:

    print(
        f"\n{direction.upper()}"
    )


    for variable in [
        "vehicles",
        "queue",
        "speed",
        "approach_arrivals"
    ]:

        values = [

            record[
                "traffic"
            ][
                direction
            ][
                variable
            ]

            for record
            in dataset
        ]


        print(

            f"{variable:18s}"

            f"min={min(values):8.2f} "

            f"max={max(values):8.2f} "

            f"mean={statistics.mean(values):8.2f}"
        )


# ============================================================
# SENSOR COVERAGE
# ============================================================

print("\n[9] SENSOR COVERAGE")

gps_counts = [

    record["gps_count"]

    for record
    in dataset
]


cctv_counts = [

    record["cctv_count"]

    for record
    in dataset
]


print(
    "GPS min/max:",
    min(gps_counts),
    "/",
    max(gps_counts)
)


print(
    "GPS mean:",
    round(
        statistics.mean(gps_counts),
        2
    )
)


print(
    "CCTV min/max:",
    min(cctv_counts),
    "/",
    max(cctv_counts)
)


print(
    "CCTV mean:",
    round(
        statistics.mean(cctv_counts),
        2
    )
)


print(
    "CCTV zero steps:",
    sum(

        1

        for count
        in cctv_counts

        if count == 0
    )
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 70
)