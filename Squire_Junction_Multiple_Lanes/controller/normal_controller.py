import traci


SUMO_BINARY = "sumo-gui.exe"
TLS_ID = "0"

traci.start([
    SUMO_BINARY,
    "-c", "../sq.sumo.cfg"
])


# ==========================================
# 1. IRC / WEBSTER BASELINE
# ==========================================

NS = 1200       # PCU/hour
EW = 2000       # PCU/hour

W_NS = 6        # metres
W_EW = 6        # metres

S_NS = 525 * W_NS
S_EW = 525 * W_EW

y_NS = NS / S_NS
y_EW = EW / S_EW

Y = y_NS + y_EW

L = 14          # total lost time

if Y < 1:
    C = (1.5 * L + 5) / (1 - Y)
    C = max(60, min(C, 120))
else:
    C = 120

green_total = C - L

g_NS = green_total * (y_NS / Y)
g_EW = green_total * (y_EW / Y)


print("\n=== NORMAL IRC/WEBSTER SIGNAL ===")
print("Y       :", round(Y, 3))
print("Cycle   :", round(C, 1), "seconds")
print("NS Green:", round(g_NS, 1), "seconds")
print("EW Green:", round(g_EW, 1), "seconds")


# ==========================================
# 2. APPLY SIGNAL TIMING
# ==========================================

program = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]

phases = program.getPhases()

phases[0].duration = g_EW
phases[1].duration = 3

phases[2].duration = g_EW
phases[3].duration = 3

phases[4].duration = g_NS
phases[5].duration = 3

phases[6].duration = g_NS
phases[7].duration = 3

program.phases = phases

traci.trafficlight.setProgramLogic(TLS_ID, program)

print("\nSignal timing applied.")


# ==========================================
# 3. DATA COLLECTION
# ==========================================

# Every unique vehicle that has entered
all_vehicles = set()

# Every vehicle that has left the network
completed_vehicles = set()

queue_history = []
speed_history = []
waiting_history = []

maximum_queue = 0


# ==========================================
# 4. RUN SIMULATION
# ==========================================

for step in range(500):

    traci.simulationStep()

    # --------------------------------------
    # Vehicles currently inside network
    # --------------------------------------

    vehicle_ids = traci.vehicle.getIDList()

    vehicle_count = len(vehicle_ids)

    # Track every unique vehicle
    all_vehicles.update(vehicle_ids)


    # --------------------------------------
    # Average speed
    # --------------------------------------

    if vehicle_ids:

        speeds = [
            traci.vehicle.getSpeed(v)
            for v in vehicle_ids
        ]

        average_speed = sum(speeds) / len(speeds)

    else:

        average_speed = 0.0


    # --------------------------------------
    # Queue
    # --------------------------------------

    queue = sum(
        1
        for v in vehicle_ids
        if traci.vehicle.getSpeed(v) < 0.5
    )

    maximum_queue = max(
        maximum_queue,
        queue
    )


    # --------------------------------------
    # Vehicles that left network
    # --------------------------------------

    departed = traci.simulation.getArrivedIDList()

    completed_vehicles.update(departed)

    flow = len(departed)


    # --------------------------------------
    # Waiting time
    # --------------------------------------

    waiting_time = sum(
        traci.vehicle.getAccumulatedWaitingTime(v)
        for v in vehicle_ids
    )


    # --------------------------------------
    # Store history
    # --------------------------------------

    queue_history.append(queue)
    speed_history.append(average_speed)
    waiting_history.append(waiting_time)


    # --------------------------------------
    # Display
    # --------------------------------------

    print(
        f"Step: {step:3d} | "
        f"Vehicles: {vehicle_count:3d} | "
        f"Queue: {queue:3d} | "
        f"Speed: {average_speed:5.2f} m/s | "
        f"Flow: {flow:3d} | "
        f"Wait: {waiting_time:6.1f} s"
    )


# ==========================================
# 5. FINAL RESULTS
# ==========================================

simulation_time = traci.simulation.getTime()

total_vehicle_count = len(all_vehicles)

maximum_queue = max(queue_history)

average_queue = (
    sum(queue_history) / len(queue_history)
    if queue_history
    else 0
)

average_speed = (
    sum(speed_history) / len(speed_history)
    if speed_history
    else 0
)

average_waiting_time = (
    sum(waiting_history) / len(waiting_history)
    if waiting_history
    else 0
)

total_completed = len(completed_vehicles)


# ==========================================
# 6. FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("       NORMAL CONTROLLER RESULTS")
print("==========================================")

print(
    "Simulation Time      :",
    round(simulation_time, 1),
    "s"
)

print(
    "Total Vehicles       :",
    total_vehicle_count
)

print(
    "Vehicles Completed   :",
    total_completed
)

print(
    "Maximum Queue        :",
    maximum_queue
)

print(
    "Average Queue        :",
    round(average_queue, 2)
)

print(
    "Average Speed        :",
    round(average_speed, 2),
    "m/s"
)

print(
    "Average Accumulated Waiting :",
    round(average_waiting_time, 2),
    "s"
)

print("==========================================")

traci.close()