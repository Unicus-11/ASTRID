import os
import sys
import torch
import torch.nn as nn
import traci

# --- Layer 1: PyTorch PINN Architecture ---
class LWR_PINN(nn.Module):
    def __init__(self, k_jam=120.0):
        super(LWR_PINN, self).__init__()
        self.k_jam = k_jam
        # Network maps (x, t) -> (Density k, Velocity v)
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        preds = self.net(inputs)
        # Apply Softplus bounds to keep density & velocity physically valid (> 0)
        k = torch.nn.functional.softplus(preds[:, 0:1])
        v = torch.nn.functional.softplus(preds[:, 1:2])
        return k, v

    def compute_physics_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        k, v = self.forward(x, t)
        q = k * v  # Flow equation q = k * v
        
        # Automatic Differentiation for LWR Equation: dk/dt + dq/dx = 0
        dk_dt = torch.autograd.grad(k, t, torch.ones_like(k), create_graph=True)[0]
        dq_dx = torch.autograd.grad(q, x, torch.ones_like(q), create_graph=True)[0]
        
        lwr_residual = dk_dt + dq_dx
        return torch.mean(lwr_residual ** 2)

# --- Layer 0: Data Ingestion & PCU Aggregation ---
PCU_WEIGHTS = {
    "motorcycle": 0.5,
    "auto_rickshaw": 0.75,
    "car": 1.0,
    "bus": 3.0,
    "vendor_cart": 1.5,
    "cow": 1.0
}

INCOMING_EDGES = ["E1", "-E2", "-E3", "-E4"]

def get_sparse_pcu_density(edge_id, sample_rate=0.20):
    """
    Extracts sparse vehicle telemetry (simulating 20% GPS probes)
    and converts counts into PCU Density (k_pcu).
    """
    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    
    # FIXED: Replaced non-existent getAdaptationTraveltime with lane length calculation
    lane_id = f"{edge_id}_0"
    try:
        edge_length_m = traci.lane.getLength(lane_id)
    except traci.exceptions.TraCIException:
        edge_length_m = 50.0  # Fallback edge length in meters

    if edge_length_m <= 0:
        edge_length_m = 50.0  
        
    sampled_pcu = 0.0
    for v_id in vehicle_ids:
        # Sparse probe filtering: process vehicle if its hash falls within sample rate
        if (hash(v_id) % 100) / 100.0 <= sample_rate:
            v_type = traci.vehicle.getTypeID(v_id)
            pcu_val = PCU_WEIGHTS.get(v_type, 1.0)
            sampled_pcu += pcu_val

    # Scale 20% probe sample back to estimated total PCU and calculate PCU/km
    estimated_total_pcu = sampled_pcu * (1.0 / sample_rate)
    k_pcu = (estimated_total_pcu / edge_length_m) * 1000.0
    return k_pcu

# --- Main TraCI Simulation & Layer 1 PINN Loop ---
def run_simulation():
    pinn = LWR_PINN(k_jam=120.0)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)

    # Launch SUMO GUI
    sumo_cmd = ["sumo-gui", "-c", "intersection.sumocfg"]
    traci.start(sumo_cmd)
    
    step = 0
    print("[A-PULSE] Simulation Started with PINN Layer 1 Estimator.")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Process Layer 1 Physics Estimation every 5 simulation seconds
        if step % 5 == 0:
            for edge in INCOMING_EDGES:
                k_pcu = get_sparse_pcu_density(edge, sample_rate=0.20)
                
                # Format tensors for PINN training pass
                x_tensor = torch.tensor([[50.0]], dtype=torch.float32)  # Mid-edge position
                t_tensor = torch.tensor([[float(step)]], dtype=torch.float32)
                target_k = torch.tensor([[k_pcu]], dtype=torch.float32)

                # Compute Losses
                optimizer.zero_grad()
                pred_k, pred_v = pinn(x_tensor, t_tensor)
                
                loss_data = torch.mean((pred_k - target_k) ** 2)
                loss_physics = pinn.compute_physics_loss(x_tensor, t_tensor)
                
                total_loss = loss_data + 0.1 * loss_physics
                total_loss.backward()
                optimizer.step()

                # Calculate LWR Shockwave Velocity (w_bf)
                v_val = pred_v.item()
                q_val = k_pcu * v_val
                k_jam = 120.0
                
                w_bf = -q_val / (k_jam - k_pcu) if (k_jam - k_pcu) > 0 else 0.0
                l_max = abs(w_bf) * 42.0  # Based on 42s red phase duration

                if step % 50 == 0:
                    print(f"[Step {step} | Edge {edge}] k_pcu: {k_pcu:.2f} | "
                          f"PINN Loss: {total_loss.item():.4f} | w_bf: {w_bf:.2f} m/s | Est L_max: {l_max:.2f}m")

    traci.close()

if __name__ == "__main__":
    run_simulation()