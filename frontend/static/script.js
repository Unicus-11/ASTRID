const socket = io();

const canvas = document.getElementById("simulationCanvas");
const ctx = canvas.getContext("2d");


// -------------------------
// START
// -------------------------

function startSimulation() {

    socket.emit("start");

}


// -------------------------
// STEP
// -------------------------

function stepSimulation() {

    socket.emit("step");

}


// -------------------------
// STOP
// -------------------------

function stopSimulation() {

    socket.emit("stop");

}


// -------------------------
// RECEIVE SUMO DATA
// -------------------------

socket.on("simulation_update", function(data) {

    document.getElementById("simTime").innerText =
        data.time.toFixed(1);

    drawSimulation(data.vehicles);

});


// -------------------------
// STATUS
// -------------------------

socket.on("status", function(data) {

    console.log(data.message);

});


// -------------------------
// DRAW
// -------------------------

function drawSimulation(vehicles) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    // Temporary background
    ctx.fillStyle = "#eeeeee";

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    // Draw vehicles
    vehicles.forEach(function(vehicle) {

        /*
         * IMPORTANT:
         *
         * SUMO coordinates are not browser
         * canvas coordinates.
         *
         * This is only a temporary conversion.
         */

        const x = vehicle.x * 5;
        const y = canvas.height - vehicle.y * 5;


        ctx.save();

        ctx.translate(x, y);

        ctx.rotate(
            -vehicle.angle * Math.PI / 180
        );


        // Vehicle
        ctx.fillStyle = "red";

        ctx.fillRect(
            -5,
            -10,
            10,
            20
        );


        ctx.restore();

    });

}