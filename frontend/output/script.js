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

let cumQueueNormal = [], cumQueueAstrid = []; // prefix sums, for O(1) "average so far" during playback

function totalQueue(frame) {
  return Object.values(frame.queues).reduce((a, v) => a + (v || 0), 0);
}

function buildPrefixSums() {
  cumQueueNormal = []; cumQueueAstrid = [];
  let sN = 0, sA = 0;
  data.normal.frames.forEach(f => { sN += totalQueue(f); cumQueueNormal.push(sN); });
  data.astrid.frames.forEach(f => { sA += totalQueue(f); cumQueueAstrid.push(sA); });
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

function renderKpiCharts() {
  const pairs = [
    ['chartWait', 'Avg Wait (s)', 'avg_wait_s'],
    ['chartSpeed', 'Avg Speed (km/h)', 'avg_speed_kmh'],
    ['chartQueue', 'Max Queue (m)', 'max_queue_m'],
    ['chartThroughput', 'Throughput (veh/hr)', 'throughput_veh_per_hr'],
  ];
  pairs.forEach(([canvasId, label, key]) => {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Normal', 'ASTRID'],
        datasets: [{
          label,
          data: [data.normal.kpis[key], data.astrid.kpis[key]],
          backgroundColor: ['#c9c9c9', '#3ddc84'],
        }],
      },
      options: {
        plugins: { legend: { display: false }, title: { display: true, text: label, color: '#e6edf3' } },
        scales: {
          x: { ticks: { color: '#e6edf3' } },
          y: { ticks: { color: '#e6edf3' }, beginAtZero: true },
        },
      },
    });
  });
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
  const queuePct = pctChange(n.avg_queue_m, a.avg_queue_m);       // lower is good
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

function lerp(a, b, t) { return a + (b - a) * t; }

// Fill color = congestion severity (blue -> yellow -> red), independent of signal phase.
function heatColor(queueM) {
  const t = Math.min(1, queueM / QUEUE_SCALE_MAX_M);
  let c1, c2, k;
  if (t < 0.5) { c1 = [59, 130, 246]; c2 = [234, 179, 8]; k = t / 0.5; }
  else { c1 = [234, 179, 8]; c2 = [220, 38, 38]; k = (t - 0.5) / 0.5; }
  return 'rgb(' + Math.round(lerp(c1[0], c2[0], k)) + ',' + Math.round(lerp(c1[1], c2[1], k)) + ',' + Math.round(lerp(c1[2], c2[2], k)) + ')';
}

// Border color = right-of-way status (separate signal, not congestion).
function phaseBorderColor(edge, stage) {
  if (!stage) return '#e0c341'; // mandatory transition phase
  return STAGE_APPROACHES[stage].includes(edge) ? '#3ddc84' : '#5a6b80';
}

function drawIntersection(canvasId, frame) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2, roadW = 60;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#14371f';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#2a2a2a';
  ctx.fillRect(0, cy - roadW / 2, w, roadW);
  ctx.fillRect(cx - roadW / 2, 0, roadW, h);

  const stage = PHASE_STAGE[frame.phase]; // undefined during a transition phase
  const lenFor = edge => Math.min(1, (frame.queues[edge] || 0) / QUEUE_SCALE_MAX_M) * (h / 2 - roadW / 2 - 10);

  const drawArm = (edge, x, y, aw, ah) => {
    ctx.fillStyle = heatColor(frame.queues[edge] || 0);
    ctx.fillRect(x, y, aw, ah);
    ctx.strokeStyle = phaseBorderColor(edge, stage);
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 1.5, y + 1.5, Math.max(aw - 3, 0), Math.max(ah - 3, 0));
  };

  // 4i = north (top), 3i = south (bottom), 1i = west (left), 2i = east (right)
  drawArm('4i', cx - roadW / 2, 0, roadW, lenFor('4i'));
  drawArm('3i', cx - roadW / 2, h - lenFor('3i'), roadW, lenFor('3i'));
  drawArm('1i', 0, cy - roadW / 2, lenFor('1i'), roadW);
  drawArm('2i', w - lenFor('2i'), cy - roadW / 2, lenFor('2i'), roadW);

  ctx.fillStyle = '#e6edf3';
  ctx.font = '11px sans-serif';
  ctx.fillText('t=' + frame.t.toFixed(0) + 's  phase=' + frame.phase + '  veh=' + frame.vehicles, 8, h - 8);
}

function renderFrame() {
  if (!data) return;
  drawIntersection('canvasNormal', data.normal.frames[frameIndex]);
  drawIntersection('canvasAstrid', data.astrid.frames[frameIndex]);
  document.getElementById('timeline').value = frameIndex;
  renderSavings(); // live, tied to current playback position -- see function docstring
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