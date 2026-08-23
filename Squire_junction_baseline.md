# Phase 1 — Building the Traffic World in SUMO

This phase is about **building and understanding the simulated traffic world** before adding any controller, sensor model, or machine-learning system.

The basic idea is:

> **SUMO defines the physical road world, the vehicles, how vehicles move through that world, and the traffic rules that govern their movement.**

---

## 1. What is SUMO?

**SUMO (Simulation of Urban MObility)** is an open-source **microscopic road traffic simulator**.

**Microscopic** means that SUMO simulates individual vehicles rather than treating traffic as one continuous mass.

For every vehicle, SUMO can determine things such as:

* position
* speed
* lane
* vehicle type
* route
* acceleration/deceleration
* interaction with other vehicles
* response to traffic signals

SUMO therefore provides the **virtual road environment** in which ASTRID operates.

### Why SUMO?

SUMO provides a common simulation framework instead of requiring every traffic-research project to build its own road network, vehicle model, routing system, traffic-light system, and simulation engine.

It is also open source, allowing researchers to modify and extend the simulator.

---

# 2. What is SiMTraM?

**SiMTraM** is an extension of SUMO developed at **IIT Bombay** for modelling **heterogeneous traffic**.

Heterogeneous traffic means different types of road users sharing the same road, rather than a road containing only identical cars.

For example:

```text
Bike + Car + Bus + HGV
```

can coexist in the same simulation.

SiMTraM adds/extends capabilities relevant to such traffic, including:

* heterogeneous vehicle behaviour
* different vehicle types
* multi-lane traffic
* lane/strip changing
* vehicle routing
* junction behaviour
* traffic simulation and visualisation

The original **Squire Junction Multi-Lane** example from SiMTraM is the starting point for this ASTRID simulation.

---

# 3. Running the Original SUMO Simulation

The main project directory is:

```bash
~/SIH/ASTRID/Squire_Junction_Multiple_Lanes
```

From the terminal, enter:

```bash
cd ~/SIH/ASTRID/Squire_Junction_Multiple_Lanes
```

Then run the SUMO GUI using the configuration file:

```bash
sumo-gui -c sq.sumo.cfg
```

The important point is that **`sq.sumo.cfg` is the entry point**.

It tells SUMO which network and traffic files to load.

---

# 4. Important SUMO Terminology

Before understanding the files, these distinctions are important.

| Term              | Meaning                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------- |
| **Network**       | The complete road system                                                                 |
| **Junction**      | Where roads meet and vehicles can change direction                                       |
| **Edge**          | A directed road segment between two junctions                                            |
| **Lane**          | One individual traffic lane belonging to an edge                                         |
| **Route**         | The sequence of edges a vehicle follows                                                  |
| **Connection**    | Defines how a lane of one edge connects to a lane of another edge                        |
| **Vehicle type**  | Defines physical and behavioural properties of a vehicle                                 |
| **Flow**          | A compact way of specifying many vehicles with common departure conditions               |
| **Traffic light** | Controls which movements/lane connections are allowed to proceed                         |
| **Incoming edge** | Edge carrying vehicles toward a junction                                                 |
| **Outgoing edge** | Edge carrying vehicles away from a junction                                              |
| **Internal lane** | Temporary lane/connection inside a junction used to model turning and crossing movements |

### Road vs Edge

In everyday language we say **road**.

SUMO usually represents a road as one or more **directed edges**.

For example:

```text
          Junction
             ↑
             │
          incoming
            edge
```

An edge may contain several lanes:

```text
Edge
├── Lane 0
├── Lane 1
└── Lane 2
```

So:

> **Edge = directed road segment**
> **Lane = individual traffic strip within that edge**

---

# 5. The Squire Junction

The network contains a central junction with four incoming approaches:

```text
                 4i
                  ↓
                  │
                  │
       1i ───── [ JUNCTION ] ───── 2i
                  │
                  │
                  ↑
                 3i
```

The actual IDs are defined by the network file.

For example:

```text
1i
2i
3i
4i
```

are incoming edges.

Each incoming edge has **three lanes**.

Therefore the junction has:

```text
4 approaches × 3 lanes
= 12 incoming lanes
```

The lanes are individual lanes within those edges.

---

# 6. Traffic Light

The central junction is controlled by a traffic light.

The network file contains a traffic-light program such as:

```xml
<tlLogic id="0" type="static" programID="0">

    <phase duration="25" state="GGrrrrGGrrrr"/>
    <phase duration="7"  state="yyrrrryyrrrr"/>
    ...
    
</tlLogic>
```

The letters describe the signal state of the controlled connections.

Generally:

```text
G = Green
y = Yellow
r = Red
```

So a phase such as:

```text
GGrrrrGGrrrr
```

means some movements are permitted while others are stopped.

The important distinction is:

> **The traffic light does not create the road or vehicle. It regulates which already-defined movements are allowed to proceed at the junction.**

---

# 7. What Does the `.net.xml` File Do?

### `sq.net.xml`

This is the **road-network definition**.

It describes the physical and traffic-control structure of the simulated world.

It contains things such as:

```text
Junctions
Edges
Lanes
Connections
Traffic lights
Lane geometry
Speed limits
Road permissions
```

For example, it defines that an edge has three lanes and a particular speed limit.

It also defines how lanes connect through the junction.

Example:

```xml
<connection
    from="2i"
    to="4o"
    fromLane="0"
    toLane="0"
    ...
/>
```

This says that a particular lane of `2i` can connect to a particular lane of `4o`.

### Therefore:

```text
sq.net.xml
       ↓
"What does the road world look like?"
```

---

# 8. How Does SUMO Know the Traffic Rules?

The important distinction is that the **network and vehicle files together describe the rules of the simulation**.

### The network defines:

* where roads exist
* which lanes exist
* which lanes can connect
* speed limits
* junction structure
* traffic-light states
* right-of-way/priority information

### Vehicle definitions define:

* vehicle size
* maximum speed
* acceleration
* deceleration
* car-following behaviour

### Routes define:

* where a particular vehicle enters
* where it travels
* where it exits

SUMO then uses its simulation models to calculate how vehicles actually move.

So the logic is approximately:

```text
ROAD STRUCTURE
      +
TRAFFIC RULES
      +
VEHICLE PROPERTIES
      +
VEHICLE ROUTE
      ↓
SUMO SIMULATION
      ↓
Vehicle movement
```

---

# 9. What Does `sq.vtype.xml` Do?

This file defines **vehicle types**.

Our simulation contains:

```text
Bike
Car
HGV
Bus
```

For example:

```xml
<vType
    id="bike"
    length="1.50"
    maxSpeed="27.77"
    ...
/>
```

The file specifies properties such as:

* length
* maximum speed
* acceleration
* deceleration
* car-following model
* other vehicle behaviour parameters

For example, our traffic distribution is approximately:

```text
50% Bike
30% Car
10% HGV
10% Bus
```

This is what makes the traffic **heterogeneous**.

---

# 10. What Does `sq.flow.xml` Do?

This file specifies **traffic demand**.

Instead of individually writing hundreds of vehicles, we can write:

```xml
<flow
    id="0"
    from="1i"
    to="54o"
    number="134"
    begin="0"
    end="3600"
    type="typedist1"
/>
```

This means, conceptually:

> Generate a specified number of vehicles during the specified time period, entering through one edge and eventually travelling toward the specified destination, using the specified vehicle-type distribution.

So:

```text
sq.flow.xml
      ↓
"How much traffic should enter?"
```

while:

```text
sq.vtype.xml
      ↓
"What kinds of vehicles are they?"
```

---

# 11. What Does `sq.rou.xml` Do?

`*.rou.xml` is the **route file used by SUMO**.

It contains the actual vehicle types and vehicle routes after the demand has been converted into individual vehicles.

For example:

```xml
<vehicle id="0.0" type="bike" ...>
    <route edges="1i 4o 54o"/>
</vehicle>
```

This tells SUMO:

```text
Vehicle 0.0
    ↓
is a bike
    ↓
enters through 1i
    ↓
travels through 4o
    ↓
continues to 54o
```

The route therefore determines **where that particular vehicle is going**.

---

# 12. What Does `sq.con.xml` Do?

`connection` means **how one lane connects to another lane**.

For example:

```xml
<connection
    from="1i"
    to="2o"
    fromLane="1"
    toLane="0"/>
```

This defines a permitted lane-to-lane connection.

This is different from a route.

### Connection

```text
Lane → Lane
```

defines **what movement is physically possible**.

### Route

```text
Edge → Edge → Edge
```

defines **where a particular vehicle chooses to travel**.

Therefore:

```text
sq.con.xml
    ↓
"What connections are possible?"

sq.rou.xml
    ↓
"Which route does this vehicle take?"
```

---

# 13. How Everything Fits Together

The entire Phase 1 world can be understood as:

```text
                  sq.net.xml
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Roads       Lanes      Traffic lights
          │           │           │
          └───────────┼───────────┘
                      ↓
                 Road network
                      │
        ┌─────────────┴─────────────┐
        ↓                           ↓
 sq.vtype.xml                 sq.flow.xml
 Vehicle properties           Traffic demand
        │                           │
        └─────────────┬─────────────┘
                      ↓
                 DUAROUTER
                      ↓
                 sq.rou.xml
                      │
                      ↓
                   SUMO
                      ↓
             Moving vehicles
```

`sq.sumo.cfg` ties the required files together and tells SUMO what to load.

---

# 14. The Most Important Concept

At this stage, think of SUMO as having four fundamental questions:

### 1. Where can vehicles travel?

**`sq.net.xml`**

```text
Roads
Lanes
Junctions
Connections
Signals
```

### 2. What are the vehicles?

**`sq.vtype.xml`**

```text
Bike
Car
HGV
Bus
```

with their physical and behavioural properties.

### 3. How much traffic is generated and where does it go?

**`sq.flow.xml`**

```text
Traffic demand
Origin
Destination
Departure period
Vehicle distribution
```

### 4. What does each individual vehicle actually do?

**`sq.rou.xml`**

```text
Vehicle
    ↓
Vehicle type
    ↓
Route
    ↓
SUMO movement
```

That is the complete foundation of **Phase 1: Building the World**.

Only after this world behaves correctly should we add the next layer: **observing the world and controlling it**.
