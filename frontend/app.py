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