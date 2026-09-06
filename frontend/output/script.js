document.getElementById('today').textContent = new Date().toDateString().toUpperCase();

// Mirrors signal_config.py's STAGE_APPROACHES / stage-per-phase mapping (read-only, for drawing only).
const STAGE_APPROACHES = { NS: ['3i', '4i'], EW: ['1i', '2i'] };
const PHASE_STAGE = { 0: 'NS', 2: 'NS', 4: 'EW', 6: 'EW' }; // transitions (1,3,5,7) -> neither
const QUEUE_SCALE_MAX_M = 150; // meters mapped to full bar length in the schematic

let data = null;
let frameIndex = 0;
let playing = false;
let playTimer = null;
const charts = {};

async function loadIndex() {
  const res = await fetch('index.json');
  const idx = await res.json();
  const select = document.getElementById('scenarioSelect');
  select.innerHTML = '';
  idx.scenarios.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    select.appendChild(opt);
  });
  if (idx.scenarios.length) loadScenario(idx.scenarios[0]);
  select.addEventListener('change', e => loadScenario(e.target.value));
}

let cumQueueNormal = [], cumQueueAstrid = [];
let cumWaitNormal = [], cumWaitAstrid = [];
let cumSpeedNormal = [], cumSpeedAstrid = [];
let cumArrivedNormal = [], cumArrivedAstrid = [];
let runningMaxQueueNormal = [], runningMaxQueueAstrid = [];

function totalQueue(frame) {
  return Object.values(frame.queues).reduce((a, v) => a + (v || 0), 0);
}

// avg_queue_m is only present in JSON produced by the current comparison.py -- fall back to
// computing it from frames directly so older logs don't produce NaN in the session summary.
function avgQueueKpi(result) {
  if (typeof result.kpis.avg_queue_m === 'number') return result.kpis.avg_queue_m;
  const frames = result.frames;
  return frames.reduce((a, f) => a + totalQueue(f), 0) / frames.length;
}

function buildPrefixSums() {
  const build = frames => {
    const cq = [], cw = [], cs = [], ca = [], rmq = [];
    let sq = 0, sw = 0, ss = 0, sa = 0, mq = 0;
    frames.forEach(f => {
      sq += totalQueue(f); cq.push(sq);
      sw += f.mean_wait_s; cw.push(sw);
      ss += f.mean_speed_mps; cs.push(ss);
      sa += (f.arrived || 0); ca.push(sa);
      mq = Math.max(mq, totalQueue(f)); rmq.push(mq);
    });
    return { cq, cw, cs, ca, rmq };
  };
  const n = build(data.normal.frames), a = build(data.astrid.frames);
  cumQueueNormal = n.cq; cumWaitNormal = n.cw; cumSpeedNormal = n.cs; cumArrivedNormal = n.ca; runningMaxQueueNormal = n.rmq;
  cumQueueAstrid = a.cq; cumWaitAstrid = a.cw; cumSpeedAstrid = a.cs; cumArrivedAstrid = a.ca; runningMaxQueueAstrid = a.rmq;
}

async function loadScenario(name) {
  const res = await fetch(name + '.json');
  data = await res.json();
  frameIndex = 0;
  const timeline = document.getElementById('timeline');
  timeline.max = data.normal.frames.length - 1;
  timeline.value = 0;
  document.getElementById('timeStart').textContent = '0s';
  document.getElementById('timeEnd').textContent = data.normal.frames[data.normal.frames.length - 1].t.toFixed(0) + 's';

  buildPrefixSums();
  renderKpiCharts();
  renderPipelineCharts();
  renderOverTimeCharts();
  renderSessionSummary();
  renderFrame();
}

// Draws the current value above each bar (e.g. "~65s"), matching the requested reference
// look. Registered per-chart below so it never touches the line charts elsewhere on the page.
const kpiValueLabelPlugin = {
  id: 'kpiValueLabel',
  afterDatasetsDraw(chart) {
    const suffix = chart.$kpiSuffix || '';
    const ctx = chart.ctx;
    const meta = chart.getDatasetMeta(0);
    meta.data.forEach((bar, i) => {
      const value = chart.data.datasets[0].data[i];
      if (value == null) return;
      ctx.save();
      ctx.fillStyle = '#e6edf3';
      ctx.font = '600 12px Inter, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('~' + Math.round(value) + suffix, bar.x, bar.y - 6);
      ctx.restore();
    });
  },
};

function renderKpiCharts() {
  const pairs = [
    ['chartWait', 'Avg Wait so far (s)', 's'],
    ['chartSpeed', 'Avg Speed so far (km/h)', 'km/h'],
    ['chartQueue', 'Max Queue so far (m)', 'm'],
    ['chartThroughput', 'Throughput so far (veh/hr)', 'veh/hr'],
  ];
  pairs.forEach(([canvasId, label, suffix]) => {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: { labels: ['Normal', 'ASTRID'], datasets: [{ label, data: [0, 0], backgroundColor: ['#c9c9c9', '#3ddc84'], borderRadius: 4, barPercentage: 0.55 }] },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 22 } },
        plugins: {
          legend: { display: false },
          title: { display: true, text: label, color: '#e6edf3', font: { size: 13, weight: '600' }, padding: { bottom: 12 } },
        },
        scales: {
          x: {
            ticks: { color: '#e6edf3', font: { size: 12 } },
            title: { display: true, text: 'Controller', color: '#9fb3c8', font: { size: 10 } },
            grid: { display: false },
          },
          y: {
            beginAtZero: true,
            ticks: { color: '#9fb3c8', font: { size: 10 }, callback: v => v + suffix },
            title: { display: true, text: suffix, color: '#9fb3c8', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
        },
      },
      plugins: [kpiValueLabelPlugin],
    });
    charts[canvasId].$kpiSuffix = suffix;
  });
}

// Recomputed from frames[0..frameIndex] only -- these bars move as playback advances,
// unlike the end-of-run Session Summary further down the page.
function updateKpiChartsLive(idx) {
  const n = idx + 1;
  const waitN = cumWaitNormal[idx] / n, waitA = cumWaitAstrid[idx] / n;
  const speedN = (cumSpeedNormal[idx] / n) * 3.6, speedA = (cumSpeedAstrid[idx] / n) * 3.6;
  const queueN = runningMaxQueueNormal[idx], queueA = runningMaxQueueAstrid[idx];
  const elapsedHours = Math.max((data.normal.frames[idx].t - data.normal.frames[0].t) / 3600, 1 / 3600);
  const thrN = cumArrivedNormal[idx] / elapsedHours, thrA = cumArrivedAstrid[idx] / elapsedHours;

  charts.chartWait.data.datasets[0].data = [waitN, waitA]; charts.chartWait.update('none');
  charts.chartSpeed.data.datasets[0].data = [speedN, speedA]; charts.chartSpeed.update('none');
  charts.chartQueue.data.datasets[0].data = [queueN, queueA]; charts.chartQueue.update('none');
  charts.chartThroughput.data.datasets[0].data = [thrN, thrA]; charts.chartThroughput.update('none');
}

function renderPipelineCharts() {
  const frames = data.astrid.frames;
  const labels = frames.map(f => f.t);
  const totals = frames.map(f => Object.values(f.queues).reduce((a, v) => a + (v || 0), 0));

  const ctx = document.getElementById('chartQueueTimeline').getContext('2d');
  if (charts.queueTimeline) charts.queueTimeline.destroy();
  charts.queueTimeline = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ data: totals, borderColor: '#7fd6ff', pointRadius: 0, borderWidth: 1 }] },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { ticks: { color: '#9fb3c8' } } },
      elements: { line: { tension: 0.2 } },
    },
  });

  const actionCanvas = document.getElementById('actionTimeline');
  const actx = actionCanvas.getContext('2d');
  actx.clearRect(0, 0, actionCanvas.width, actionCanvas.height);
  const w = actionCanvas.width / frames.length;
  frames.forEach((f, i) => {
    actx.fillStyle = f.action === 'REQUEST_NEXT' ? '#ff9d7a' : '#3d6fdc';
    actx.fillRect(i * w, 0, Math.max(w, 1), actionCanvas.height);
  });

  document.getElementById('reqCount').textContent = data.astrid.kpis.requested_transitions;
  document.getElementById('forcedCount').textContent = data.astrid.kpis.forced_transitions;
}

// Illustrative annualized savings estimate -- assumptions are documented in the dashboard's
// "assumptions-note" text and adjustable here. Not a measured/calibrated figure.
const METERS_PER_VEHICLE = 7;
const IDLE_FUEL_L_PER_HOUR = 0.6;
const FUEL_PRICE_PER_L = 100;      // currency units per liter
const CO2_KG_PER_LITER = 2.31;
const OPERATING_HOURS_PER_DAY = 16;
const DAYS_PER_YEAR = 365;

function renderSavings() {
  if (!data) return;
  const n = frameIndex + 1;
  const avgNormal = cumQueueNormal[frameIndex] / n;
  const avgAstrid = cumQueueAstrid[frameIndex] / n;
  const queueReductionPct = avgNormal > 0 ? Math.max(0, (1 - avgAstrid / avgNormal) * 100) : 0;

  const avgQueueDiffM = Math.max(0, avgNormal - avgAstrid);
  const vehiclesSaved = avgQueueDiffM / METERS_PER_VEHICLE;
  const litersPerYear = vehiclesSaved * IDLE_FUEL_L_PER_HOUR * OPERATING_HOURS_PER_DAY * DAYS_PER_YEAR;
  const moneyPerYear = litersPerYear * FUEL_PRICE_PER_L;
  const co2TonsPerYear = (litersPerYear * CO2_KG_PER_LITER) / 1000;

  document.getElementById('fuelSaving').textContent = '\u20B9' + (moneyPerYear / 1e6).toFixed(2) + 'M / Year';
  document.getElementById('fuelSavingSub').textContent = '(' + litersPerYear.toFixed(0) + ' L/yr, ' + queueReductionPct.toFixed(0) + '% lower queue)';
  document.getElementById('emissionSaving').textContent = co2TonsPerYear.toFixed(1) + ' Metric Tons CO\u2082/Year';
  document.getElementById('emissionSavingSub').textContent = '(' + queueReductionPct.toFixed(0) + '% lower queue, est.)';
}

function renderOverTimeCharts() {
  const labels = data.normal.frames.map(f => f.t);
  const queueNormal = data.normal.frames.map(totalQueue);
  const queueAstrid = data.astrid.frames.map(totalQueue);
  const waitNormal = data.normal.frames.map(f => f.mean_wait_s);
  const waitAstrid = data.astrid.frames.map(f => f.mean_wait_s);

  const ROLL = 60; // seconds
  const rollingThroughput = frames => frames.map((_, i) => {
    const start = Math.max(0, i - ROLL + 1);
    const sum = frames.slice(start, i + 1).reduce((a, f) => a + (f.arrived || 0), 0);
    const windowS = i - start + 1;
    return (sum / windowS) * 3600;
  });
  const thrNormal = rollingThroughput(data.normal.frames);
  const thrAstrid = rollingThroughput(data.astrid.frames);

  const lineOpts = () => ({
    plugins: { legend: { labels: { color: '#e6edf3' } } },
    scales: { x: { display: false }, y: { ticks: { color: '#9fb3c8' } } },
    elements: { point: { radius: 0 }, line: { borderWidth: 1.5, tension: 0.15 } },
  });

  const mk = (id, key, ln, la) => {
    const ctx = document.getElementById(id).getContext('2d');
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets: [
        { label: 'Normal', data: ln, borderColor: '#b39ddb' },
        { label: 'ASTRID', data: la, borderColor: '#4fd1ff' },
      ] },
      options: lineOpts(),
    });
  };
  mk('overQueue', 'queue', queueNormal, queueAstrid);
  mk('overDelay', 'wait', waitNormal, waitAstrid);
  mk('overThroughput', 'thr', thrNormal, thrAstrid);
}

function pctChange(normalVal, astridVal) {
  if (normalVal === 0) return 0;
  return ((astridVal - normalVal) / normalVal) * 100;
}

function renderSessionSummary() {
  const n = data.normal.kpis, a = data.astrid.kpis;
  const queuePct = pctChange(avgQueueKpi(data.normal), avgQueueKpi(data.astrid)); // lower is good
  const waitPct = pctChange(n.avg_wait_s, a.avg_wait_s);          // lower is good
  const speedPct = pctChange(n.avg_speed_kmh, a.avg_speed_kmh);   // higher is good
  const thrPct = pctChange(n.throughput_veh_per_hr, a.throughput_veh_per_hr); // higher is good

  const setArrow = (id, pct, higherIsGood) => {
    const el = document.getElementById(id);
    const good = higherIsGood ? pct > 0 : pct < 0;
    const arrow = pct > 0 ? '\u25B2' : '\u25BC';
    el.textContent = arrow + ' ' + Math.abs(pct).toFixed(0) + '%';
    el.className = 'summary-arrow ' + (pct > 0 ? 'up-' : 'down-') + (good ? 'good' : 'bad');
  };
  setArrow('sumQueueArrow', queuePct, false);
  setArrow('sumWaitArrow', waitPct, false);
  setArrow('sumSpeedArrow', speedPct, true);
  setArrow('sumThroughputArrow', thrPct, true);

  const goodCount = [queuePct < 0, waitPct < 0, speedPct > 0, thrPct > 0].filter(Boolean).length;
  const verdictEl = document.getElementById('summaryVerdict');
  if (goodCount >= 3) { verdictEl.textContent = 'IMPROVED'; verdictEl.className = 'summary-verdict improved'; }
  else if (goodCount <= 1) { verdictEl.textContent = 'WORSE'; verdictEl.className = 'summary-verdict worse'; }
  else { verdictEl.textContent = 'MIXED RESULT \u2014 SEE KPIs ABOVE'; verdictEl.className = 'summary-verdict mixed'; }
}

// Border color = right-of-way status (separate signal from congestion visuals).
function phaseBorderColor(edge, stage) {
  if (!stage) return '#e0c341'; // mandatory transition phase
  return STAGE_APPROACHES[stage].includes(edge) ? '#3ddc84' : '#5a6b80';
}

// Vehicle shapes, sized (meters) from the project's own vType definitions -- not invented.
const VEHICLE_TYPES = {
  bike: { length: 1.5, width: 1.0, color: '#f2c744' },
  car: { length: 4.5, width: 3.0, color: '#4fd1ff' },
  hgv: { length: 10.21, width: 5.0, color: '#ff9d7a' },
  bus: { length: 11.54, width: 5.0, color: '#7fffb0' },
};
const PIXELS_PER_METER = 3;

const LANES = 3;               // matches signal_config.py: 3 lanes per approach
const LANE_W = 20;
const ROAD_W = LANES * LANE_W;
const VEH_LEN = 10, VEH_GAP = 3;
const MAX_VEH_PER_LANE = 10;   // fallback-mode visual cap so a huge queue doesn't overflow the canvas

// Right-of-way status colors, used as the vehicle's outline/highlight (real-data mode, where
// fill = vehicle type) or as the vehicle's fill itself (fallback mode, no type data available).
const VEH_COLOR_MOVING = '#6fff7e';
const VEH_COLOR_WAITING = '#ff6b6b';
const VEH_COLOR_TRANSITION = '#e0c341';

function statusColorFor(edge, stage) {
  const served = stage ? STAGE_APPROACHES[stage].includes(edge) : false;
  return !stage ? VEH_COLOR_TRANSITION : (served ? VEH_COLOR_MOVING : VEH_COLOR_WAITING);
}

function roundedRectPath(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// Draws one vehicle as a rounded rect with a small "windshield" highlight near the front
// (the end closest to the intersection), and an optional colored outline for status.
function drawVehicle(ctx, cx, cy, vw, vh, axis, alongDir, fillColor, outlineColor) {
  const x = cx - vw / 2, y = cy - vh / 2;
  roundedRectPath(ctx, x, y, vw, vh, 2);
  ctx.fillStyle = fillColor;
  ctx.fill();
  if (outlineColor) { ctx.strokeStyle = outlineColor; ctx.lineWidth = 1; ctx.stroke(); }

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  const hl = 3;
  if (axis === 'vertical') {
    const hy = alongDir.y < 0 ? y : y + vh - hl;
    ctx.fillRect(x + 1, hy, vw - 2, hl);
  } else {
    const hx = alongDir.x < 0 ? x : x + vw - hl;
    ctx.fillRect(hx, y + 1, hl, vh - 2);
  }
}

// Renders REAL per-vehicle positions from frame.queue_vehicles (type, lane, dist_to_stop_m) --
// fill = vehicle type (bike/car/hgv/bus, sized from the scenario's own vType lengths), outline
// = right-of-way status. This is what makes cars visibly change lanes / advance toward the
// stop line frame to frame.
function drawArmVehiclesReal(ctx, edge, frame, origin, laneStep, alongDir, axis, stage) {
  const outline = statusColorFor(edge, stage);
  frame.queue_vehicles.filter(v => v.edge === edge).forEach(v => {
    const spec = VEHICLE_TYPES[v.type] || VEHICLE_TYPES.car;
    const dist = v.dist_to_stop_m * PIXELS_PER_METER;
    const cx = origin.x + laneStep.x * v.lane + alongDir.x * dist;
    const cy = origin.y + laneStep.y * v.lane + alongDir.y * dist;
    const lenPx = spec.length * PIXELS_PER_METER, widPx = spec.width * PIXELS_PER_METER;
    const vw = axis === 'vertical' ? widPx : lenPx;
    const vh = axis === 'vertical' ? lenPx : widPx;
    drawVehicle(ctx, cx, cy, vw, vh, axis, alongDir, spec.color, outline);
  });
}

// Fallback for JSON logged before queue_vehicles existed: approximate count from queue meters,
// fill = right-of-way status (no type data available yet).
function drawArmVehiclesApprox(ctx, edge, frame, origin, laneStep, alongDir, axis, stage) {
  const color = statusColorFor(edge, stage);
  const vehCount = Math.min(Math.round((frame.queues[edge] || 0) / METERS_PER_VEHICLE), LANES * MAX_VEH_PER_LANE);
  for (let i = 0; i < vehCount; i++) {
    const lane = i % LANES;
    const posInLane = Math.floor(i / LANES);
    const dist = posInLane * (VEH_LEN + VEH_GAP) + VEH_LEN / 2 + VEH_GAP;
    const cx = origin.x + laneStep.x * lane + alongDir.x * dist;
    const cy = origin.y + laneStep.y * lane + alongDir.y * dist;
    const vw = axis === 'vertical' ? LANE_W - 6 : VEH_LEN;
    const vh = axis === 'vertical' ? VEH_LEN : LANE_W - 6;
    drawVehicle(ctx, cx, cy, vw, vh, axis, alongDir, color, null);
  }
}

function drawArmVehicles(ctx, edge, frame, origin, laneStep, alongDir, axis) {
  const stage = PHASE_STAGE[frame.phase];
  if (frame.queue_vehicles) drawArmVehiclesReal(ctx, edge, frame, origin, laneStep, alongDir, axis, stage);
  else drawArmVehiclesApprox(ctx, edge, frame, origin, laneStep, alongDir, axis, stage);
}

function drawLaneMarkings(ctx, cx, cy, w, h) {
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.setLineDash([6, 6]);
  ctx.lineWidth = 1;
  for (let l = 1; l < LANES; l++) {
    ctx.beginPath(); ctx.moveTo(cx - ROAD_W / 2 + l * LANE_W, 0); ctx.lineTo(cx - ROAD_W / 2 + l * LANE_W, cy - ROAD_W / 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - ROAD_W / 2 + l * LANE_W, cy + ROAD_W / 2); ctx.lineTo(cx - ROAD_W / 2 + l * LANE_W, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy - ROAD_W / 2 + l * LANE_W); ctx.lineTo(cx - ROAD_W / 2, cy - ROAD_W / 2 + l * LANE_W); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + ROAD_W / 2, cy - ROAD_W / 2 + l * LANE_W); ctx.lineTo(w, cy - ROAD_W / 2 + l * LANE_W); ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawIntersection(canvasId, frame) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0e2b17';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#2a2a2a';
  ctx.fillRect(0, cy - ROAD_W / 2, w, ROAD_W);
  ctx.fillRect(cx - ROAD_W / 2, 0, ROAD_W, h);
  drawLaneMarkings(ctx, cx, cy, w, h);

  const stage = PHASE_STAGE[frame.phase];
  // Right-of-way border tint on the whole arm, congestion severity communicated by how many
  // vehicle rectangles are queued rather than a solid color block (see the on-canvas legend).
  const armColor = edge => phaseBorderColor(edge, stage);
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = armColor('4i'); ctx.fillRect(cx - ROAD_W / 2, 0, ROAD_W, cy - ROAD_W / 2);
  ctx.fillStyle = armColor('3i'); ctx.fillRect(cx - ROAD_W / 2, cy + ROAD_W / 2, ROAD_W, h - (cy + ROAD_W / 2));
  ctx.fillStyle = armColor('1i'); ctx.fillRect(0, cy - ROAD_W / 2, cx - ROAD_W / 2, ROAD_W);
  ctx.fillStyle = armColor('2i'); ctx.fillRect(cx + ROAD_W / 2, cy - ROAD_W / 2, w - (cx + ROAD_W / 2), ROAD_W);
  ctx.globalAlpha = 1;

  // 4i = north (top): lanes left-to-right, vehicles queue upward from the stop line.
  drawArmVehicles(ctx, '4i', frame, { x: cx - ROAD_W / 2 + LANE_W / 2, y: cy - ROAD_W / 2 }, { x: LANE_W, y: 0 }, { x: 0, y: -1 }, 'vertical');
  // 3i = south (bottom): vehicles queue downward.
  drawArmVehicles(ctx, '3i', frame, { x: cx - ROAD_W / 2 + LANE_W / 2, y: cy + ROAD_W / 2 }, { x: LANE_W, y: 0 }, { x: 0, y: 1 }, 'vertical');
  // 1i = west (left): lanes top-to-bottom, vehicles queue leftward.
  drawArmVehicles(ctx, '1i', frame, { x: cx - ROAD_W / 2, y: cy - ROAD_W / 2 + LANE_W / 2 }, { x: 0, y: LANE_W }, { x: -1, y: 0 }, 'horizontal');
  // 2i = east (right): vehicles queue rightward.
  drawArmVehicles(ctx, '2i', frame, { x: cx + ROAD_W / 2, y: cy - ROAD_W / 2 + LANE_W / 2 }, { x: 0, y: LANE_W }, { x: 1, y: 0 }, 'horizontal');

  ctx.fillStyle = '#e6edf3';
  ctx.font = '11px sans-serif';
  ctx.fillText('t=' + frame.t.toFixed(0) + 's  phase=' + frame.phase + '  veh=' + frame.vehicles, 8, h - 8);
}

function renderFrame() {
  if (!data) return;
  drawIntersection('canvasNormal', data.normal.frames[frameIndex]);
  drawIntersection('canvasAstrid', data.astrid.frames[frameIndex]);
  document.getElementById('timeline').value = frameIndex;
  renderSavings();           // live, tied to current playback position
  updateKpiChartsLive(frameIndex); // live, tied to current playback position
}

document.getElementById('timeline').addEventListener('input', e => {
  frameIndex = parseInt(e.target.value, 10);
  renderFrame();
});

document.getElementById('speed').addEventListener('input', e => {
  document.getElementById('speedVal').textContent = e.target.value + 'x';
  if (playing) startPlay();
});

function step() {
  if (!data) return;
  frameIndex = Math.min(frameIndex + 1, data.normal.frames.length - 1);
  renderFrame();
  if (frameIndex >= data.normal.frames.length - 1) pause();
}

function startPlay() {
  clearInterval(playTimer);
  const speed = parseInt(document.getElementById('speed').value, 10);
  const intervalMs = Math.max(10, 200 / speed);
  playTimer = setInterval(step, intervalMs);
  playing = true;
}
function pause() { clearInterval(playTimer); playing = false; }

document.getElementById('btnPlay').addEventListener('click', startPlay);
document.getElementById('btnPause').addEventListener('click', pause);
document.getElementById('btnStep').addEventListener('click', () => { pause(); step(); });

loadIndex();