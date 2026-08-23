
# ASTRID — Squire Junction Multi-Lane Simulation

ASTRID uses **SUMO (Simulation of Urban MObility)** to simulate traffic at the Squire Junction multi-lane network.

The project has been updated to run successfully with:

- **SUMO 1.27.1**
- **Python 3.x**

---

# 1. Project Structure

The main SUMO files are:

```text
Squire_Junction_Multiple_Lanes/
│
├── sq.sumo.cfg       → SUMO configuration
├── sq.net.xml        → Road network
├── sq.rou.xml        → Vehicles and routes
│
└── controller/
    ├── sensor_simulator.py
    ├── state_extractor.py
    └── dataset_validator.py
````

The basic flow is:

```text
SUMO Network
     +
Vehicle Routes
     ↓
SUMO Simulation
     ↓
sensor_simulator.py
     ↓
Sensor Dataset
     ↓
dataset_validator.py
     ↓
ML / State Estimation
```

---

# 2. Running the Simulation

Move into the project directory:

```bash
cd ~/SIH/ASTRID/Squire_Junction_Multiple_Lanes
```

Start SUMO-GUI:

```bash
sumo-gui -c sq.sumo.cfg
```

SUMO-GUI allows us to:

* Start and pause the simulation
* Change simulation speed
* Watch vehicles move
* Inspect vehicles
* Observe different vehicle types
* Observe lane usage
* Inspect the junction

For easier observation, set the simulation time around **550 seconds** and change the visualization mode from **Standard** to **Real World**.

---

# 3. Simulation Configuration

The main configuration file is:

```text
sq.sumo.cfg
```

It connects the simulation components:

```text
sq.sumo.cfg
      │
      ├── sq.net.xml
      │       ↓
      │   Road network
      │
      └── sq.rou.xml
              ↓
        Vehicles + routes
```

The simulation runs from:

```text
0 seconds → 3600 seconds
```

Therefore, the complete simulation represents:

```text
3600 seconds = 1 hour
```

The converted simulation successfully completed:

```text
Simulation ended at time: 3600.00
```

It inserted:

```text
1600 vehicles
```

There were only three vehicle teleports caused by detected collisions. These are simulation-behavior issues and are separate from the XML conversion.

---

# 4. Vehicle Types

The simulation contains four vehicle types:

```xml
<vType id="bike" .../>
<vType id="car" .../>
<vType id="hgv" .../>
<vType id="bus" .../>
```

| Vehicle | SUMO ID | Maximum Speed |  Length | Width | Share |
| ------- | ------- | ------------: | ------: | ----: | ----: |
| Bike    | `bike`  |     27.77 m/s |   1.5 m |   1 m |   50% |
| Car     | `car`   |      25.0 m/s |   4.5 m |   3 m |   30% |
| HGV     | `hgv`   |     19.44 m/s | 10.21 m |   5 m |   10% |
| Bus     | `bus`   |     19.44 m/s | 11.54 m |   5 m |   10% |

The distribution is:

```text
50%  Bikes
30%  Cars
10%  HGVs
10%  Buses
----------------
100% Total
```

For example, out of approximately 100 vehicles:

```text
50 bikes
30 cars
10 HGVs
10 buses
```

The actual numbers can vary because the simulation generates individual vehicles from the traffic demand.

---

# 5. Vehicle Behaviour

The four vehicle types use the **Krauss car-following model**.

Important parameters:

```text
sigma = 0.0
tau   = 1.0
```

These parameters describe how vehicles follow the vehicle in front of them.

The vehicle type is available to our sensor system through SUMO:

```python
traci.vehicle.getTypeID(vehicle_id)
```

Therefore our dataset can contain:

```json
"vehicle_type": "bike"
```

or:

```json
"vehicle_type": "car"
```

or:

```json
"vehicle_type": "hgv"
```

or:

```json
"vehicle_type": "bus"
```

This is important because ASTRID does **not** treat every vehicle as identical.

---

# 6. SUMO Edges and the Intersection

SUMO represents roads as **edges**.

An edge is simply a road segment.

For our junction, the main incoming edges are:

```text
1i
2i
3i
4i
```

The `i` means:

```text
incoming
```

These roads lead **toward the intersection**.

The outgoing edges include:

```text
1o
2o
3o
4o
```

The `o` means:

```text
outgoing
```

These roads lead **away from the intersection**.

A simplified view is:

```text
                    4i
                    ↓
                    │
                    │
             1i → [ JUNCTION ] → 2o
                    │
                    │
                    ↑
                    3i
```

The numbers (`1`, `2`, `3`, `4`) are simply the IDs used by this SUMO network.

They are not mathematical values.

---

# 7. Route vs Movement

A vehicle's **route** is the complete sequence of road edges that it will travel through.

Example:

```text
1i → 4o → 54o
```

This means:

```text
1i
 ↓
[JUNCTION]
 ↓
4o
 ↓
54o
```

Here:

* `1i` = road before the junction
* `4o` = road immediately after the junction
* `54o` = another road segment farther along the route

---

# 8. What Does "Downstream" Mean?

**Downstream** simply means:

> farther along the vehicle's direction of travel.

For example:

```text
A → B → C → D
```

If a vehicle is moving from A to D:

```text
A = upstream
B = downstream of A
C = further downstream
D = even further downstream
```

Therefore:

```text
1i → 4o → 54o
```

has:

```text
1i  →  4o  →  54o
       ↑
       |
   immediately
   after junction
```

`54o` is farther downstream than `4o`.

---

# 9. Movement

A **route** tells us all the roads the vehicle will use.

A **movement** tells us the vehicle's higher-level movement through the junction.

For example:

```text
Route:

1i → 4o → 54o
```

For the junction, we only need:

```text
1i → 4o
```

Therefore the movement is:

```text
west_to_north
```

We use:

```python
incoming = route[0]
outgoing = route[1]
```

because:

```python
route[0] = "1i"
route[1] = "4o"
route[2] = "54o"
```

So:

```text
route:
1i → 4o → 54o

movement:
1i → 4o
```

This prevents a downstream edge such as `54o` from incorrectly changing the movement label.

---

# 10. Why Movement Is Ground Truth

Our sensors do **not** directly know the complete future route.

The simulated ground truth does.

Therefore:

```text
SUMO
 │
 ├── route
 │
 └── movement
        ↓
   Ground Truth
```

While the simulated sensors observe only:

```text
GPS
 ├── position
 ├── speed
 ├── edge
 └── lane

CCTV
 ├── position
 ├── speed
 ├── edge
 ├── lane
 └── vehicle type
```

Later, the ML model will learn:

```text
Sensor observations
        ↓
   State estimator
        ↓
Estimated hidden state
        ↓
Movement prediction
```

The SUMO movement is therefore used as the **training label**, not as sensor input.

This is important because otherwise we would be giving the neural network the answer.

---

# 11. Current Dataset Architecture

Our data is separated into three parts:

```text
                    SUMO
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
       Ground      GPS        CCTV
       Truth
          │          │          │
          │          │          │
          └──────────┼──────────┘
                     ↓
             sensor_dataset.json
```

### Ground Truth

Contains information that SUMO knows exactly:

```text
position
speed
edge
lane
route
route_index
movement
vehicle_type
```

### GPS

Represents sparse probe-vehicle observations:

```text
position
speed
edge
lane
vehicle_type
timestamp
```

### CCTV

Represents local camera observations:

```text
camera_id
position
speed
edge
lane
lane_position
vehicle_type
timestamp
```

The important principle is:

```text
Sensors → observations

SUMO → ground truth
```

The neural network will later learn to reconstruct hidden traffic state from the observations.

---

# 12. Current Development Order

We are intentionally building the system in stages:

```text
1. SUMO simulation
       ↓
2. Sensor simulation
       ↓
3. JSON dataset
       ↓
4. Dataset validation
       ↓
5. Inspect and correct data
       ↓
6. State representation
       ↓
7. ML state estimator
       ↓
8. Traffic prediction
       ↓
9. Control
```

We are currently around **step 4**.

The immediate goal is **not yet to train the neural network**.

First we need to make sure that the dataset is structurally and logically correct.

```


