"""


RUN : DISHA@LAPTOP-JO1S4POA MINGW64 ~/SIH/ASTRID (main)
$ python scripts/validate_dataset.py datasets/scenario_0001/sensor_dataset.json  
============================================================
ASTRID DATASET VALIDATOR
============================================================

Purpose
-------
Validate sensor_dataset.json files produced by ASTRID.

The validator checks:

    1. Dataset structure
    2. Simulation timestamps
    3. Traffic-state values
    4. Queue consistency
    5. Ground-truth consistency
    6. GPS/CCTV values
    7. Traffic-state ↔ ground-truth relationship

IMPORTANT
---------
traffic["direction"]["vehicles"] represents vehicles currently
on the four incoming approach edges.

ground_truth represents ALL vehicles currently present in SUMO.

Therefore:

    traffic vehicle count != ground_truth count

is NOT an error.

============================================================
"""

import json
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_DURATION = 3600

APPROACHES = {
    "north",
    "south",
    "east",
    "west",
}

VALID_VEHICLE_TYPES = {
    "bike",
    "car",
    "bus",
    "hgv",
}


# ============================================================
# HELPERS
# ============================================================

def fail(errors, message):
    errors.append(message)


def check_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(dataset_path):
    errors = []
    warnings = []

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found:")
        print(f"       {dataset_path}")
        return False

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"ERROR: Could not read JSON:")
        print(f"       {e}")
        return False

    # --------------------------------------------------------
    # BASIC STRUCTURE
    # --------------------------------------------------------

    if not isinstance(data, list):
        fail(errors, "Dataset root must be a list.")

    if not data:
        fail(errors, "Dataset is empty.")

    if errors:
        print_errors(errors)
        return False

    print("=" * 60)
    print("ASTRID DATASET VALIDATION")
    print("=" * 60)

    print(f"Dataset : {dataset_path}")
    print(f"Records : {len(data)}")
    print()

    # --------------------------------------------------------
    # RECORD COUNT
    # --------------------------------------------------------

    if len(data) != EXPECTED_DURATION:
        fail(
            errors,
            f"Expected {EXPECTED_DURATION} records, "
            f"found {len(data)}."
        )

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    expected_time = 1.0

    for index, record in enumerate(data):

        timestamp = record.get("simulation_time")

        if not check_number(timestamp):
            fail(
                errors,
                f"Record {index}: invalid simulation_time."
            )
            continue

        if timestamp != expected_time:
            fail(
                errors,
                f"Record {index}: expected time "
                f"{expected_time}, found {timestamp}."
            )

        expected_time += 1.0

    # --------------------------------------------------------
    # RECORD VALIDATION
    # --------------------------------------------------------

    for index, record in enumerate(data):

        prefix = f"Record {index} (t={record.get('simulation_time')})"

        # ====================================================
        # TRAFFIC
        # ====================================================

        traffic = record.get("traffic")

        if not isinstance(traffic, dict):
            fail(errors, f"{prefix}: missing traffic.")
            continue

        missing = APPROACHES - set(traffic.keys())

        if missing:
            fail(
                errors,
                f"{prefix}: missing approaches: "
                f"{sorted(missing)}"
            )

        for approach in APPROACHES:

            if approach not in traffic:
                continue

            state = traffic[approach]

            if not isinstance(state, dict):
                fail(
                    errors,
                    f"{prefix}: {approach} traffic state "
                    f"is not an object."
                )
                continue

            # ----------------------------------------------
            # VEHICLE COUNT
            # ----------------------------------------------

            vehicles = state.get("vehicles")

            if not isinstance(vehicles, int) or vehicles < 0:
                fail(
                    errors,
                    f"{prefix}: {approach}.vehicles "
                    f"must be a non-negative integer."
                )

            # ----------------------------------------------
            # QUEUE
            # ----------------------------------------------

            queue = state.get("queue")

            if not isinstance(queue, int) or queue < 0:
                fail(
                    errors,
                    f"{prefix}: {approach}.queue "
                    f"must be a non-negative integer."
                )

            elif isinstance(vehicles, int) and queue > vehicles:
                fail(
                    errors,
                    f"{prefix}: {approach}.queue "
                    f"({queue}) > vehicles ({vehicles})."
                )

            # ----------------------------------------------
            # SPEED
            # ----------------------------------------------

            speed = state.get("average_speed")

            if speed is not None:

                if not check_number(speed):
                    fail(
                        errors,
                        f"{prefix}: {approach}.average_speed "
                        f"must be numeric."
                    )

                elif speed < 0:
                    fail(
                        errors,
                        f"{prefix}: {approach}.average_speed "
                        f"cannot be negative."
                    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    total_ground_truth = 0
    unique_vehicle_ids = set()

    for index, record in enumerate(data):

        ground_truth = record.get("sensors", {}).get(
            "ground_truth"
        )

        if not isinstance(ground_truth, list):
            fail(
                errors,
                f"Record {index}: ground_truth is missing "
                f"or is not a list."
            )
            continue

        total_ground_truth += len(ground_truth)

        for vehicle in ground_truth:

            vehicle_id = vehicle.get("id")

            if not vehicle_id:
                fail(
                    errors,
                    f"Record {index}: ground-truth vehicle "
                    f"has no ID."
                )
                continue

            unique_vehicle_ids.add(vehicle_id)

            # ----------------------------------------------
            # SPEED
            # ----------------------------------------------

            speed = vehicle.get("speed")

            if not check_number(speed):
                fail(
                    errors,
                    f"Record {index}: vehicle {vehicle_id} "
                    f"has invalid speed."
                )

            elif speed < 0:
                fail(
                    errors,
                    f"Record {index}: vehicle {vehicle_id} "
                    f"has negative speed."
                )

            # ----------------------------------------------
            # VEHICLE TYPE
            # ----------------------------------------------

            vehicle_type = vehicle.get("vehicle_type")

            if (
                vehicle_type is not None
                and vehicle_type not in VALID_VEHICLE_TYPES
            ):
                warnings.append(
                    f"Record {index}: unknown vehicle type "
                    f"'{vehicle_type}'."
                )

    # ========================================================
    # GPS / CCTV
    # ========================================================

    for index, record in enumerate(data):

        gps_count = record.get("gps_count")
        cctv_count = record.get("cctv_count")

        if not isinstance(gps_count, int) or gps_count < 0:
            fail(
                errors,
                f"Record {index}: invalid gps_count."
            )

        if not isinstance(cctv_count, int) or cctv_count < 0:
            fail(
                errors,
                f"Record {index}: invalid cctv_count."
            )

    # ========================================================
    # REPORT
    # ========================================================

    print("Ground-truth observations :", total_ground_truth)
    print("Unique vehicles observed  :", len(unique_vehicle_ids))
    print()

    if warnings:
        print("WARNINGS")
        print("-" * 60)

        for warning in warnings[:20]:
            print("WARNING:", warning)

        if len(warnings) > 20:
            print(
                f"... and {len(warnings) - 20} more warnings."
            )

        print()

    if errors:
        print_errors(errors)
        return False

    print("RESULT")
    print("-" * 60)
    print("PASS: Dataset passed all validation checks.")
    print()
    print("NOTE:")
    print(
        "Traffic vehicle counts and ground-truth counts are "
        "not expected to be equal."
    )
    print(
        "Traffic counts describe vehicles on incoming "
        "approach edges."
    )
    print(
        "Ground truth describes all vehicles currently "
        "present in SUMO."
    )

    return True


# ============================================================
# ERROR REPORT
# ============================================================

def print_errors(errors):

    print("ERRORS")
    print("-" * 60)

    for error in errors[:50]:
        print("ERROR:", error)

    if len(errors) > 50:
        print(
            f"... and {len(errors) - 50} more errors."
        )

    print()


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "    python scripts/validate_dataset.py "
            "datasets/scenario_0002/sensor_dataset.json"
        )

        sys.exit(1)

    dataset_path = sys.argv[1]

    success = validate_dataset(dataset_path)

    sys.exit(0 if success else 1)