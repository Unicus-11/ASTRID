import json
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCENARIO_FILE = BASE_DIR / "scenarios" / "baseline.json"
FLOW_FILE = BASE_DIR / "sq.flow.xml"


# ============================================================
# LOAD SCENARIO
# ============================================================

with open(SCENARIO_FILE, "r") as f:
    scenario = json.load(f)

print("Loaded scenario:", scenario["name"])

simulation_end = scenario["simulation_end"]
vehicle_demand = scenario["vehicle_demand"]

print("Vehicle demand:", vehicle_demand)


# ============================================================
# VALIDATE VEHICLE DISTRIBUTION
# ============================================================

total_probability = sum(vehicle_demand.values())

if abs(total_probability - 1.0) > 0.001:
    raise ValueError(
        f"Vehicle demand must sum to 1.0, "
        f"but got {total_probability}"
    )


# ============================================================
# FLOW CONFIGURATION
# ============================================================

# Each incoming road has three possible movements:
#
# incoming -> outgoing
#
# 12 total flows:
# 4 incoming roads × 3 movements

movements = [
    ("1i", "54o"),
    ("1i", "52o"),
    ("1i", "53o"),

    ("2i", "51o"),
    ("2i", "53o"),
    ("2i", "54o"),

    ("3i", "54o"),
    ("3i", "52o"),
    ("3i", "51o"),

    ("4i", "51o"),
    ("4i", "53o"),
    ("4i", "52o"),
]


# ============================================================
# TOTAL VEHICLES
# ============================================================

TOTAL_VEHICLES = 1600


# ============================================================
# DISTRIBUTE VEHICLES ACROSS MOVEMENTS
# ============================================================

base_flow = TOTAL_VEHICLES // len(movements)

remainder = TOTAL_VEHICLES % len(movements)

flow_numbers = []

for i in range(len(movements)):

    number = base_flow

    if i < remainder:
        number += 1

    flow_numbers.append(number)


print("\nGenerated flow counts:")

for i, number in enumerate(flow_numbers):
    print(f"Flow {i}: {number}")


print(
    "Total:",
    sum(flow_numbers)
)


# ============================================================
# WRITE FLOW XML
# ============================================================

xml_lines = []

xml_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
xml_lines.append("<routes>")


# ============================================================
# CREATE FLOWS
# ============================================================

for i, ((from_edge, to_edge), number) in enumerate(
    zip(movements, flow_numbers)
):

    xml_lines.append(
        f'    <flow '
        f'id="{i}" '
        f'from="{from_edge}" '
        f'to="{to_edge}" '
        f'number="{number}" '
        f'begin="0" '
        f'end="{simulation_end}" '
        f'type="typedist1" '
        f'departLane="free" '
        f'departSpeed="random" />'
    )


xml_lines.append("</routes>")


# ============================================================
# SAVE
# ============================================================

with open(FLOW_FILE, "w") as f:
    f.write("\n".join(xml_lines))


print("\nFlow file generated:")
print(FLOW_FILE)