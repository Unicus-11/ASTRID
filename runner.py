import os
import sys
from pathlib import Path

# Ensure SUMO_HOME environment variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # TraCI API to interact with SUMO

# Base path targeting your local directory
base_dir = Path(__file__).resolve().parent / "Squire_Junction_Multiple_Lanes"

# Config and output files inside the directory
cfg_file = base_dir / "sq.sumo.cfg"
rou_file = base_dir / "sq.rou.xml"

def run_simulation():
    # Command to execute SUMO with GUI
    sumo_cmd = ["sumo-gui", "-c", str(cfg_file)]
    
    # Start TraCI simulation
    traci.start(sumo_cmd)
    step = 0
    
    # Run simulation loop (3600 seconds as configured in sq.sumo.cfg)
    while step < 3600:
        traci.simulationStep()
        
        # Example: Print active vehicle count every 100 steps
        if step % 100 == 0:
            vehicle_count = traci.vehicle.getIDCount()
            print(f"Step {step}: {vehicle_count} active vehicles on road.")
            
        step += 1

    traci.close()
    print("Simulation completed successfully.")

if __name__ == "__main__":
    run_simulation()