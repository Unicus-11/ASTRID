import traci
from sensor_simulator import get_sensor_data


SUMO_BINARY = "sumo-gui"

traci.start([
    SUMO_BINARY,
    "-c", "sq.sumo.cfg"
])


TLS_ID = "0"

# ==========================================
# TRAFFIC LIGHT: CONTROLLED LINKS
# ==========================================

links = traci.trafficlight.getControlledLinks(TLS_ID)

for i, link in enumerate(links):
    print(i, "->", link)


# Keep SUMO alive briefly
traci.simulationStep()


# ==========================================
# INITIALIZE
# ==========================================

previous_vehicle_count = 0


for step in range(100):  # ===> Change the range here to increase simulation time

    traci.simulationStep()

    # ==========================================
    # 1. GROUND TRUTH
    # ==========================================

    vehicle_ids = traci.vehicle.getIDList()

    vehicle_count = len(vehicle_ids)

    if vehicle_ids:

        speeds = [
            traci.vehicle.getSpeed(vehicle_id)
            for vehicle_id in vehicle_ids
        ]

        average_speed = sum(speeds) / len(speeds)

        queue = sum(
            1
            for vehicle_id in vehicle_ids
            if traci.vehicle.getSpeed(vehicle_id) < 0.5
        )

    else:

        average_speed = 0.0
        queue = 0

    flow = max(
        vehicle_count - previous_vehicle_count,
        0
    )

    previous_vehicle_count = vehicle_count

    # ==========================================
    # 2. STATE
    # ==========================================

    state = {
        "vehicles": vehicle_count,
        "queue": queue,
        "speed": round(average_speed, 2),
        "flow": flow
    }

    # ==========================================
    # 3. SIMULATED SENSORS
    # ==========================================

    sensor_data = get_sensor_data()

    gps_count = len(sensor_data["gps"])
    cctv_count = len(sensor_data["cctv"])

    # ==========================================
    # 4. DISPLAY
    # ==========================================

    print(
        f"Step: {step:3d} | "
        f"Truth: {state['vehicles']:3d} | "
        f"Queue: {state['queue']:3d} | "
        f"Speed: {state['speed']:5.2f} m/s | "
        f"Flow: {state['flow']:2d} | "
        f"CCTV: {cctv_count:3d} | "
        f"GPS: {gps_count:2d}"
    )


traci.close()