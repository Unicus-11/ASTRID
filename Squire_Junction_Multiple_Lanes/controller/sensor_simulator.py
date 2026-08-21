import traci
import random

GPS_PENETRATION = 0.20


def get_sensor_data():

    vehicle_ids = traci.vehicle.getIDList()

    # -----------------------------
    # GPS / Connected Vehicle Data
    # -----------------------------
    gps_vehicles = []

    for vehicle_id in vehicle_ids:

        if random.random() < GPS_PENETRATION:

            gps_vehicles.append({
                "id": vehicle_id,
                "speed": traci.vehicle.getSpeed(vehicle_id),
                "position": traci.vehicle.getPosition(vehicle_id)
            })

    # -----------------------------
    # Limited CCTV Coverage
    # -----------------------------
    cctv_vehicles = []

    for vehicle_id in vehicle_ids:

        x, y = traci.vehicle.getPosition(vehicle_id)

        # Temporary camera coverage area
        # around the intersection
        if -100 <= x <= 100 and -100 <= y <= 100:
            cctv_vehicles.append(vehicle_id)

    return {
        "ground_truth": len(vehicle_ids),
        "gps": gps_vehicles,
        "cctv": cctv_vehicles
    }