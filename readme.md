# SUMO Visualization and Running the Multi-Lane Simulation

This project uses **SUMO (Simulation of Urban MObility)** to simulate traffic through the Squire Junction multi-lane network.

The legacy SUMO project was originally created using much older SUMO/XML conventions. It has now been converted so that it runs successfully with **SUMO 1.27.1**.

Requirement:
- SUMO 1.27.1
- Python 3.x

## Running the Simulation

First, open a terminal and move into the multi-lane project directory:

```bash
cd ~/SIH/ASTRID/Squire_Junction_Multiple_Lanes
```

Then launch the **SUMO graphical interface**:

```bash
sumo-gui -c sq.sumo.cfg
```

This opens the SUMO visualization window.

Inside SUMO-GUI, you can:

* Press **Play** to run the simulation. Edit the time to 550 to see better and change drop menu from standard to real world
* Pause and resume the simulation.
* Adjust the simulation speed.
* Watch vehicles move through the junction.
* Observe different vehicle types and lane usage.
* Inspect individual vehicles and network elements.

The command uses:

```text
sq.sumo.cfg
    │
    ├── sq.net.xml   → road network
    │
    └── sq.rou.xml   → vehicle routes and traffic demand
```

The simulation is configured to run from **0 to 3600 seconds (one hour)**.

---

## Vehicle Types and Traffic Demand

The original project used a legacy vehicle-type distribution called:

```xml
<vtypeDistribution id="typedist1">
```

The distribution contains four vehicle categories.

| Vehicle | ID     | Maximum Speed |  Length | Width | Probability |
| ------- | ------ | ------------: | ------: | ----: | ----------: |
| Bike    | `bike` |     27.77 m/s |   1.5 m |   1 m |         50% |
| Car     | `car`  |      25.0 m/s |   4.5 m |   3 m |         30% |
| HGV     | `hgv`  |     19.44 m/s | 10.21 m |   5 m |         10% |
| Bus     | `bus`  |     19.44 m/s | 11.54 m |   5 m |         10% |

The probabilities sum to:

```text
0.5 + 0.3 + 0.1 + 0.1 = 1.0
```

Therefore, when the original traffic demand uses `typedist1`, vehicles are distributed approximately as:

```text
50% bikes
30% cars
10% HGVs
10% buses
```

All four vehicle types originally used the **Krauss car-following model**, with:

```text
sigma = 0.0
tau   = 1.0
```

### Important conversion detail

The original legacy files referenced the distribution directly:

```xml
type="typedist1"
```

During the modernization process, `duarouter` converts that demand into individual vehicles in `sq.rou.xml`. The generated route file now contains concrete vehicle types such as:

```xml
<vType id="bike" .../>
<vType id="car" .../>
<vType id="hgv" .../>
<vType id="bus" .../>
```

and individual vehicles reference those types:

```xml
<vehicle id="0.0" type="bike" ...>
```

This is why the current `sq.sumo.cfg` does **not** need to load `sq.vtype.xml` again.

---

## Current Simulation Result

The converted model has successfully completed a full one-hour simulation:

```text
Simulation ended at time: 3600.00.
```

It successfully inserted:

```text
1600 vehicles
```

There were only three vehicle teleports caused by detected collisions. These are **simulation-behavior issues**, not XML conversion failures, and can be investigated separately.
