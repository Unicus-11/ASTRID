"""

To achieve side-by-side split visualization without relying on heavy SUMO GUI embedded inside `iframe` tags, standard web applications decouple the SUMO simulation backend from the UI rendering layer.

SUMO native GUI cannot run inside web browsers natively. Instead, your Python backend runs the SUMO engine headlessly and streams frame state updates directly to a WebGL/Canvas frontend via WebSockets.

---

### Recommended Architecture & Tech Stack

* **Frontend Framework**: **React.js** (or Vue.js) paired with **Tailwind CSS** or **Sass** for UI layout controls and side panels.
* **2D/3D Rendering Layer**: **HTML5 Canvas (2D API)**, **PixiJS**, or **Three.js (WebGL)**.
* **Real-time Communication**: **WebSockets** (using `socket.io` or `websockets` in Python).
* **Backend Orchestrator**: Python (FastAPI / Flask) driving SUMO via **TraCI** or **libsumo**.

---

### Step-by-Step Implementation Strategy

**1. Parse Road Network Topology Once**

* Convert your `sq.net.xml` road network into a light JSON structure containing junction points, lane polylines, stop lines, and signal locations.
* Load this JSON on the React frontend to draw static junction background assets once (roads, lane dividers, signal indicators).

**2. Backend State Streaming Loop**

* Run two concurrent SUMO instances via TraCI:
* **Instance A**: Default/Fixed-Time Controller.
* **Instance B**: ASTRID AI Controller.


* At every simulation step (`traci.simulationStep()`), query vehicle positions `(x, y)`, orientation `angle`, type `vtype`, and signal states.
* Broadcast this frame payload over WebSockets:

```json
{
  "step": 120,
  "normal": {
    "signals": ["G", "r", "r"],
    "vehicles": [{"id": "v1", "x": 105.2, "y": 42.1, "angle": 90, "type": "car"}]
  },
  "astrid": {
    "signals": ["G", "G", "G"],
    "vehicles": [{"id": "v1", "x": 140.8, "y": 42.1, "angle": 90, "type": "car"}]
  }
}

```

**3. Frontend Canvas Dual-Viewport Rendering**

* Create two dedicated `<canvas>` elements inside your React layout (one for **Normal**, one for **ASTRID**).
* On receiving each frame payload, clear and redraw vehicle objects and traffic signal color indicators using coordinates supplied by SUMO.
* Use `requestAnimationFrame` for smooth position interpolation between updates.

---

### Technology Choices Comparison

| Tool/Library | Feasibility | Best For |
| --- | --- | --- |
| **Streamlit** | Low | Quick data dashboards; struggles with high-FPS dynamic Canvas rendering. |
| **Leaflet / OpenLayers** | Medium | Map-overlay applications; less suited for high-frequency custom agent animation. |
| **React + HTML5 Canvas / PixiJS** | **High (Recommended)** | Smooth 60 FPS dual-view animations with custom UI dashboards & metrics integration. |
| **React + Three.js** | **High** | 3D visual perspectives (similar to the 3D gridlock rendering shown in your sample layout). |

---

### Pre-Rendering / Playback Alternative

If real-time bidirectional control is not strictly required during frontend playback:

1. Run your scenarios off-line.
2. Store frame-by-frame JSON logs or vector state traces.
3. Stream or load pre-recorded JSON arrays directly into the React canvas timeline controller for interactive speed adjustments (1x, 50x, 100x).


"""






import os
import traci
from flask import Flask, render_template, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "sumo",
        "Squire_Junction_Multiple_Lanes",
        "sq.sumo.cfg"
    )
)

running = False


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():

    global running

    if running:
        return jsonify({
            "status": "already_running"
        })

    if not os.path.exists(CONFIG_PATH):
        return jsonify({
            "status": "error",
            "message": f"SUMO config not found:\n{CONFIG_PATH}"
        }), 400

    try:

        print("Starting SUMO-GUI...")
        print("Config:", CONFIG_PATH)

        # IMPORTANT:
        # traci.start() automatically starts SUMO-GUI
        # and creates the TraCI connection.
        traci.start([
            "sumo-gui",
            "-c",
            CONFIG_PATH
        ])

        running = True

        return jsonify({
            "status": "started"
        })

    except Exception as e:

        print("SUMO ERROR:", e)

        try:
            traci.close()
        except:
            pass

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/step", methods=["POST"])
def step():

    if not running:
        return jsonify({
            "status": "not_running"
        })

    try:

        traci.simulationStep()

        return jsonify({
            "status": "stepped",
            "time": traci.simulation.getTime()
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/stop", methods=["POST"])
def stop():

    global running

    try:

        if running:
            traci.close()

        running = False

        return jsonify({
            "status": "stopped"
        })

    except Exception as e:

        running = False

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )