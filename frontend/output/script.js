document.getElementById('today').textContent = new Date().toDateString().toUpperCase();

const charts = {};
let scenario = null;
let simEndS = null;

// Running state for the currently loaded scenario. We do NOT keep a full
// frames[] array (there is no upfront length any more -- this is live) --
// only what's needed for cumulative KPIs plus the small arrays the line
// charts append to as data streams in.
let normalHist = { t: [], queue: [], wait: [], speed: [], arrived: [] };
let astridHist = { t: [], queue: [], wait: [], speed: [], arrived: [] };
let running = null; // see resetRunning()

function resetRunning() {
  running = {
    n: 0,
    sumQueueN: 0, sumQueueA: 0,
    sumWaitN: 0, sumWaitA: 0,
    sumSpeedN: 0, sumSpeedA: 0,
    arrivedN: 0, arrivedA: 0,
    maxQueueN: 0, maxQueueA: 0,
    firstT: null, lastT: null,
  };
  normalHist = { t: [], queue: [], wait: [], speed: [], arrived: [] };
  astridHist = { t: [], queue: [], wait: [], speed: [], arrived: [] };
}
resetRunning();

function totalQueue(frame) {
  const q = frame.queues || {};
  return Object.values(q).reduce((a, v) => a + (v || 0), 0);
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(wsProto + '//' + location.host + '/ws');

function setConnLabel(text) {
  document.getElementById('connLabel').textContent = text;
}

ws.addEventListener('open', () => {
  setConnLabel('LIVE');
  fetch('/api/scenarios').then(r => r.json()).then(data => {
    const select = document.getElementById('scenarioSelect');
    select.innerHTML = '';
    (data.scenarios || []).forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      select.appendChild(opt);
    });
    if (!data.scenarios || !data.scenarios.length) {
      showScenarioError('No scenarios found under the configured SCENARIOS_ROOT.');
      return;
    }
    select.addEventListener('change', e => loadScenario(e.target.value));
    loadScenario(data.scenarios[0]);
  }).catch(err => showScenarioError('Could not load scenario list: ' + err.message));
});

ws.addEventListener('close', () => setConnLabel('DISCONNECTED'));
ws.addEventListener('error', () => setConnLabel('ERROR'));

ws.addEventListener('message', ev => {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }

  if (msg.type === 'scenario_loaded') {
    scenario = msg.name;
    simEndS = msg.sim_end_s;
    resetRunning();
    document.getElementById('timeline').max = simEndS || 0;
    document.getElementById('timeline').value = 0;
    document.getElementById('timeline').disabled = false;
    document.getElementById('timeStart').textContent = '0s';
    document.getElementById('timeEnd').textContent = (simEndS ?? '--') + 's';
    showScenarioError(null);
    renderKpiCharts();
    renderOverTimeChartsInit();
    renderPipelineChartsInit();
    document.getElementById('reqCount').textContent = '0';
    document.getElementById('forcedCount').textContent = '0';
  } else if (msg.type === 'frame') {
    onFrame(msg);
  } else if (msg.type === 'done') {
    setConnLabel('LIVE (finished)');
  } else if (msg.type === 'error') {
    showScenarioError(msg.message);
  }
});

function showScenarioError(m) {
  const el = document.getElementById('scenarioError');
  if (!m) { el.style.display = 'none'; el.textContent = ''; return; }
  el.style.display = 'block';
  el.textContent = m;
  console.error('[dashboard] ' + m);
}

function loadScenario(name) {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ cmd: 'load_scenario', name }));
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------
function onFrame(msg) {
  const nf = msg.normal, af = msg.astrid;
  const r = running;

  const qN = totalQueue(nf), qA = totalQueue(af);
  r.n += 1;
  r.sumQueueN += qN; r.sumQueueA += qA;
  r.sumWaitN += (nf.mean_wait_s || 0); r.sumWaitA += (af.mean_wait_s || 0);
  r.sumSpeedN += (nf.mean_speed_mps || 0); r.sumSpeedA += (af.mean_speed_mps || 0);
  r.arrivedN += (nf.arrived || 0); r.arrivedA += (af.arrived || 0);
  r.maxQueueN = Math.max(r.maxQueueN, qN); r.maxQueueA = Math.max(r.maxQueueA, qA);
  if (r.firstT === null) r.firstT = nf.t;
  r.lastT = nf.t;

  normalHist.t.push(nf.t); normalHist.queue.push(qN); normalHist.wait.push(nf.mean_wait_s || 0);
  normalHist.speed.push(nf.mean_speed_mps || 0); normalHist.arrived.push(nf.arrived || 0);
  astridHist.t.push(af.t); astridHist.queue.push(qA); astridHist.wait.push(af.mean_wait_s || 0);
  astridHist.speed.push(af.mean_speed_mps || 0); astridHist.arrived.push(af.arrived || 0);

  document.getElementById('timeline').value = nf.t;
  document.getElementById('reqCount').textContent = msg.requested_transitions;
  document.getElementById('forcedCount').textContent = msg.forced_transitions;

  updateKpiChartsLive();
  updateOverTimeCharts();
  updatePipelineCharts(af);
  renderSavings();
  renderSessionSummary();
}

// ---------------------------------------------------------------------------
// KPI bar charts (Normal vs ASTRID, running average/max "so far")
// ---------------------------------------------------------------------------
const kpiValueLabelPlugin = {
  id: 'kpiValueLabel',
  afterDatasetsDraw(chart) {
    const suffix = chart.$kpiSuffix || '';
    const ctx = chart.ctx;
    const meta = chart.getDatasetMeta(0);
    meta.data.forEach((bar, i) => {
      const value = chart.data.datasets[0].data[i];
      if (value == null || Number.isNaN(value)) return;
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
      data: { labels: ['Normal', 'ASTRID (RF)'], datasets: [{ label, data: [0, 0], backgroundColor: ['#c9c9c9', '#3ddc84'], borderRadius: 4, barPercentage: 0.55 }] },
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

function updateKpiChartsLive() {
  const r = running;
  if (r.n === 0) return;
  const waitN = r.sumWaitN / r.n, waitA = r.sumWaitA / r.n;
  const speedN = (r.sumSpeedN / r.n) * 3.6, speedA = (r.sumSpeedA / r.n) * 3.6;
  const elapsedHours = Math.max((r.lastT - r.firstT) / 3600, 1 / 3600);
  const thrN = r.arrivedN / elapsedHours, thrA = r.arrivedA / elapsedHours;

  charts.chartWait.data.datasets[0].data = [waitN, waitA]; charts.chartWait.update('none');
  charts.chartSpeed.data.datasets[0].data = [speedN, speedA]; charts.chartSpeed.update('none');
  charts.chartQueue.data.datasets[0].data = [r.maxQueueN, r.maxQueueA]; charts.chartQueue.update('none');
  charts.chartThroughput.data.datasets[0].data = [thrN, thrA]; charts.chartThroughput.update('none');
}

// ---------------------------------------------------------------------------
// Pipeline charts (ASTRID queue timeline + action timeline strip)
// ---------------------------------------------------------------------------
function renderPipelineChartsInit() {
  const ctx = document.getElementById('chartQueueTimeline').getContext('2d');
  if (charts.queueTimeline) charts.queueTimeline.destroy();
  charts.queueTimeline = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: '#7fd6ff', pointRadius: 0, borderWidth: 1 }] },
    options: {
      animation: false,
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { ticks: { color: '#9fb3c8' } } },
      elements: { line: { tension: 0.2 } },
    },
  });

  const actionCanvas = document.getElementById('actionTimeline');
  const actx = actionCanvas.getContext('2d');
  actx.clearRect(0, 0, actionCanvas.width, actionCanvas.height);
  actionCanvas.$pxPerFrame = simEndS ? actionCanvas.width / simEndS : 0.1;
}

function updatePipelineCharts(astridFrame) {
  const qc = charts.queueTimeline;
  qc.data.labels.push(astridFrame.t);
  qc.data.datasets[0].data.push(totalQueue(astridFrame));
  qc.update('none');

  const actionCanvas = document.getElementById('actionTimeline');
  const actx = actionCanvas.getContext('2d');
  const w = Math.max(actionCanvas.$pxPerFrame || 0.1, 1);
  const x = astridFrame.t * (actionCanvas.$pxPerFrame || 0.1);
  const isRequest = astridFrame.resolved === 'BEGIN_TRANSITION' || astridFrame.resolved === 'FORCE_TRANSITION_MAX_GREEN';
  actx.fillStyle = isRequest ? '#ff9d7a' : '#3d6fdc';
  actx.fillRect(x, 0, w, actionCanvas.height);
}

// ---------------------------------------------------------------------------
// Savings estimate (illustrative, unchanged assumptions)
// ---------------------------------------------------------------------------
const METERS_PER_VEHICLE = 7;
const IDLE_FUEL_L_PER_HOUR = 0.6;
const FUEL_PRICE_PER_L = 100;
const CO2_KG_PER_LITER = 2.31;
const OPERATING_HOURS_PER_DAY = 16;
const DAYS_PER_YEAR = 365;

function renderSavings() {
  const r = running;
  if (r.n === 0) return;
  const avgNormal = r.sumQueueN / r.n;
  const avgAstrid = r.sumQueueA / r.n;
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

// ---------------------------------------------------------------------------
// Over-time line charts (grow live as frames stream in)
// ---------------------------------------------------------------------------
const ROLL = 60; // seconds, rolling throughput window

function rollingThroughput(hist) {
  const n = hist.arrived.length;
  const start = Math.max(0, n - ROLL);
  const sum = hist.arrived.slice(start).reduce((a, v) => a + v, 0);
  const windowS = n - start;
  return windowS > 0 ? (sum / windowS) * 3600 : 0;
}

function renderOverTimeChartsInit() {
  const lineOpts = () => ({
    animation: false,
    plugins: { legend: { labels: { color: '#e6edf3' } } },
    scales: { x: { display: false }, y: { ticks: { color: '#9fb3c8' } } },
    elements: { point: { radius: 0 }, line: { borderWidth: 1.5, tension: 0.15 } },
  });
  const mk = id => {
    const ctx = document.getElementById(id).getContext('2d');
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Normal', data: [], borderColor: '#b39ddb' },
        { label: 'ASTRID (RF)', data: [], borderColor: '#4fd1ff' },
      ] },
      options: lineOpts(),
    });
  };
  mk('overQueue'); mk('overDelay'); mk('overThroughput');
}

function updateOverTimeCharts() {
  const t = normalHist.t[normalHist.t.length - 1];
  ['overQueue', 'overDelay', 'overThroughput'].forEach(id => charts[id].data.labels.push(t));

  charts.overQueue.data.datasets[0].data.push(normalHist.queue[normalHist.queue.length - 1]);
  charts.overQueue.data.datasets[1].data.push(astridHist.queue[astridHist.queue.length - 1]);
  charts.overQueue.update('none');

  charts.overDelay.data.datasets[0].data.push(normalHist.wait[normalHist.wait.length - 1]);
  charts.overDelay.data.datasets[1].data.push(astridHist.wait[astridHist.wait.length - 1]);
  charts.overDelay.update('none');

  charts.overThroughput.data.datasets[0].data.push(rollingThroughput(normalHist));
  charts.overThroughput.data.datasets[1].data.push(rollingThroughput(astridHist));
  charts.overThroughput.update('none');
}

// ---------------------------------------------------------------------------
// Session summary ("so far", updates live -- there is no fixed "final" frame
// until the episode reports done)
// ---------------------------------------------------------------------------
function pctChange(normalVal, astridVal) {
  if (!Number.isFinite(normalVal) || !Number.isFinite(astridVal) || normalVal === 0) return null;
  return ((astridVal - normalVal) / normalVal) * 100;
}

function renderSessionSummary() {
  const r = running;
  if (r.n === 0) return;
  const elapsedHours = Math.max((r.lastT - r.firstT) / 3600, 1 / 3600);

  const queuePct = pctChange(r.sumQueueN / r.n, r.sumQueueA / r.n);
  const waitPct = pctChange(r.sumWaitN / r.n, r.sumWaitA / r.n);
  const speedPct = pctChange(r.sumSpeedN / r.n, r.sumSpeedA / r.n);
  const thrPct = pctChange(r.arrivedN / elapsedHours, r.arrivedA / elapsedHours);

  const setArrow = (id, pct, higherIsGood) => {
    const el = document.getElementById(id);
    if (pct === null) { el.textContent = '--'; el.className = 'summary-arrow'; return; }
    const good = higherIsGood ? pct > 0 : pct < 0;
    const arrow = pct > 0 ? '\u25B2' : '\u25BC';
    el.textContent = arrow + ' ' + Math.abs(pct).toFixed(0) + '%';
    el.className = 'summary-arrow ' + (pct > 0 ? 'up-' : 'down-') + (good ? 'good' : 'bad');
  };
  setArrow('sumQueueArrow', queuePct, false);
  setArrow('sumWaitArrow', waitPct, false);
  setArrow('sumSpeedArrow', speedPct, true);
  setArrow('sumThroughputArrow', thrPct, true);

  const results = [
    queuePct !== null ? queuePct < 0 : null,
    waitPct !== null ? waitPct < 0 : null,
    speedPct !== null ? speedPct > 0 : null,
    thrPct !== null ? thrPct > 0 : null,
  ].filter(v => v !== null);
  const goodCount = results.filter(Boolean).length;
  const verdictEl = document.getElementById('summaryVerdict');
  if (!results.length) { verdictEl.textContent = 'INSUFFICIENT KPI DATA'; verdictEl.className = 'summary-verdict mixed'; }
  else if (goodCount / results.length >= 0.75) { verdictEl.textContent = 'ASTRID IMPROVED'; verdictEl.className = 'summary-verdict improved'; }
  else if (goodCount / results.length <= 0.25) { verdictEl.textContent = 'ASTRID WORSE'; verdictEl.className = 'summary-verdict worse'; }
  else { verdictEl.textContent = 'MIXED RESULT \u2014 SEE KPIs ABOVE'; verdictEl.className = 'summary-verdict mixed'; }
}

// ---------------------------------------------------------------------------
// Playback controls -> bridge commands (lockstep is enforced server-side)
// ---------------------------------------------------------------------------
document.getElementById('btnPlay').addEventListener('click', () => ws.send(JSON.stringify({ cmd: 'play' })));
document.getElementById('btnPause').addEventListener('click', () => ws.send(JSON.stringify({ cmd: 'pause' })));
document.getElementById('btnStep').addEventListener('click', () => ws.send(JSON.stringify({ cmd: 'step' })));
document.getElementById('speed').addEventListener('input', e => {
  document.getElementById('speedVal').textContent = e.target.value + 'x';
  ws.send(JSON.stringify({ cmd: 'set_speed', value: parseInt(e.target.value, 10) }));
});
// The timeline slider is now a live, read-only progress indicator (it moves
// as frames arrive) -- scrubbing it does nothing, since this drives two real
// sumo-gui processes, not a static array you can seek in from the browser.