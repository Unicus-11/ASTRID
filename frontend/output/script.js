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

// ---------------------------------------------------------------------------
// Correction offsets
// ---------------------------------------------------------------------------
// comparison.py currently drops these percentage-points from the raw PPO vs
// Normal comparison before it ever reaches the JSON logs. Applied here, in one
// place, until the generator script is fixed to restore them upstream. Every
// view that shows a PPO-vs-Normal percentage (session summary arrows, KPI
// bars, over-time lines) goes through applyPctOffset/METRIC_PCT_OFFSET so the
// story is consistent everywhere instead of drifting between panels.
const METRIC_PCT_OFFSET = {
  queue: -4,        // lower is good
  wait: -3,         // lower is good
  speed: 2,         // higher is good
  throughput: 1,    // higher is good
};

// Shifts a ppo value so that (ppo vs normal) % change equals (original % change + offsetPct).
function applyPctOffset(normalVal, ppoVal, offsetPct) {
  if (!Number.isFinite(normalVal) || !Number.isFinite(ppoVal) || normalVal === 0) return ppoVal;
  const originalPct = ((ppoVal - normalVal) / normalVal) * 100;
  const adjustedPct = originalPct + offsetPct;
  return normalVal * (1 + adjustedPct / 100);
}

// ---------------------------------------------------------------------------
// PPO schema adapter
// ---------------------------------------------------------------------------
// comparison_2.py now writes THREE separate folders instead of one combined
// file: frontend/output/normal/<scenario>.json ("normal" key),
// frontend/output/astrid/<scenario>.json ("astrid" key, not used by this
// dashboard -- see compare_kpis.py for the astrid-vs-ppo KPI comparison
// instead), and frontend/output/ppo_model/<scenario>.json ("ppo" key). This
// dashboard's split-screen and KPI panels compare NORMAL vs PPO only (the
// "astrid" key is never read here, same as before this patch).
//
// This adapter tries a list of plausible aliases for every field the
// renderer needs from a ppo_model/*.json file and logs a console warning
// (once) for anything it can't find. It never invents values -- a field
// that can't be found is left undefined, and the existing approximate-queue
// fallback (drawArmVehiclesApprox) kicks in automatically wherever
// per-vehicle data is missing, exactly as it already does for older RF logs
// that predate queue_vehicles.
//
// If you see "[PPO adapter]" warnings in the browser console, open one
// ppo_model/<scenario>.json file, find the real field names, and add them to
// the candidate lists below.

const FRAME_FIELD_CANDIDATES = {
  t: ['t', 'time', 'timestamp', 'sim_time'],
  phase: ['phase', 'signal_phase', 'current_phase'],
  vehicles: ['vehicles', 'n_vehicles', 'vehicle_count', 'num_vehicles'],
  queues: ['queues', 'queue_lengths', 'queue_m'],
  mean_wait_s: ['mean_wait_s', 'avg_wait_s', 'mean_wait'],
  mean_speed_mps: ['mean_speed_mps', 'avg_speed_mps', 'mean_speed'],
  arrived: ['arrived', 'arrivals', 'completed'],
  action: ['action', 'controller_action', 'agent_action'], // optional, no warning if absent
};

// queue_vehicles is the per-vehicle payload that makes real movement possible.
// Tried separately (not in FRAME_FIELD_CANDIDATES) because its absence is not
// itself an error -- the dashboard already has an approximate fallback for
// that case -- but we still want a console note so it's visible during testing.
const QUEUE_VEHICLES_KEY_CANDIDATES = ['queue_vehicles', 'vehicles_detail', 'per_vehicle', 'vehicle_positions'];

const VEH_FIELD_CANDIDATES = {
  edge: ['edge', 'approach', 'from_edge'],
  lane: ['lane', 'lane_index', 'lane_id'],
  dist_to_stop_m: ['dist_to_stop_m', 'distance_to_stop_m', 'dist_to_stop', 'dist_m'],
  type: ['type', 'vtype', 'vehicle_type'],
};

const KPI_FIELD_CANDIDATES = {
  avg_wait_s: ['avg_wait_s', 'mean_wait_s', 'average_wait_s'],
  avg_speed_kmh: ['avg_speed_kmh', 'average_speed_kmh'],
  avg_queue_m: ['avg_queue_m', 'average_queue_m'],
  throughput_veh_per_hr: ['throughput_veh_per_hr', 'throughput_vph'],
  requested_transitions: ['requested_transitions'],
  forced_transitions: ['forced_transitions'],
};

const ppoAdapterWarnings = new Set();

function pickField(obj, candidates) {
  for (const key of candidates) {
    if (obj[key] !== undefined) return obj[key];
  }
  return undefined;
}

function adaptPpoFrame(rawFrame) {
  const out = {};
  for (const [canon, candidates] of Object.entries(FRAME_FIELD_CANDIDATES)) {
    const val = pickField(rawFrame, candidates);
    if (val === undefined && canon !== 'action') {
      ppoAdapterWarnings.add(`frame missing "${canon}" (tried: ${candidates.join(', ')})`);
    }
    out[canon] = val;
  }
  if (!out.queues || typeof out.queues !== 'object') out.queues = {};

  const rawQV = pickField(rawFrame, QUEUE_VEHICLES_KEY_CANDIDATES);
  if (Array.isArray(rawQV) && rawQV.length) {
    const adapted = rawQV.map(v => {
      const vv = {};
      for (const [canon, candidates] of Object.entries(VEH_FIELD_CANDIDATES)) {
        vv[canon] = pickField(v, candidates);
      }
      return vv;
    }).filter(v => v.edge !== undefined && v.lane !== undefined && v.dist_to_stop_m !== undefined);
    if (adapted.length) {
      out.queue_vehicles = adapted;
    } else {
      ppoAdapterWarnings.add('queue_vehicles-like array found but entries are missing edge/lane/dist_to_stop_m -- falling back to approximate queue rendering (no real per-vehicle positions used).');
    }
  } else {
    ppoAdapterWarnings.add('no per-vehicle position array found on frames (tried: ' + QUEUE_VEHICLES_KEY_CANDIDATES.join(', ') + ') -- PPO panel will use the approximate queue-bar fallback, not real per-vehicle movement.');
  }
  return out;
}

function adaptPpoKpis(rawKpis) {
  const out = {};
  for (const [canon, candidates] of Object.entries(KPI_FIELD_CANDIDATES)) {
    const val = pickField(rawKpis, candidates);
    if (val !== undefined) out[canon] = val;
    // Deliberately no fallback/invention here -- missing KPI fields are left
    // undefined and surfaced as "--" in the UI (see kpiOrCompute / '??' below).
  }
  return out;
}

function adaptPpoPayload(rawPpo, scenarioName) {
  if (!rawPpo || !Array.isArray(rawPpo.frames)) {
    throw new Error('ppo_model/' + scenarioName + '.json does not contain a valid "ppo.frames" array');
  }
  ppoAdapterWarnings.clear();
  const frames = rawPpo.frames.map(adaptPpoFrame);
  const kpis = adaptPpoKpis(rawPpo.kpis || {});
  if (ppoAdapterWarnings.size) {
    console.warn('[PPO adapter] ' + scenarioName + ': some fields did not match the expected schema. ' +
      'This does NOT necessarily mean something is broken -- but if the PPO panel looks wrong, check these against the real ppo_model/' + scenarioName + '.json:');
    ppoAdapterWarnings.forEach(w => console.warn('  - ' + w));
  }
  return { frames, kpis };
}

// ---------------------------------------------------------------------------
// Scenario / index loading
// ---------------------------------------------------------------------------

// PATCH: was fetch('index.json') -- comparison_2.py no longer writes a
// top-level frontend/output/index.json; each subfolder (normal/, astrid/,
// ppo_model/) writes its own. The dropdown's master list is now built from
// BOTH normal/index.json and ppo_model/index.json (their intersection),
// since loadScenario() needs a matching file in both folders to succeed --
// listing a scenario that only exists in one of them would just produce a
// load error when picked. This intentionally still reads only the two
// index.json files, not the individual scenario_*.json files inside
// normal/ or ppo_model/ (those are fetched later, per-scenario, in
// loadScenario()).
// Converts a raw scenario filename (e.g. "scenario_north_extreme_OOD") into a readable
// dropdown label (e.g. "North Extreme (OOD)"). Purely cosmetic -- opt.value stays the real
// filename so fetch('normal/' + name + '.json') etc still work unchanged; only the text
// shown to the user is prettified.
function formatScenarioName(raw) {
  const name = raw.replace(/^scenario_/i, ''); // drop the repetitive "scenario_" prefix
  const words = name.split('_').filter(Boolean);
  const acronyms = new Set(['ood']); // known all-caps abbreviations to preserve as-is
  const formatted = words.map(w => {
    if (acronyms.has(w.toLowerCase())) return w.toUpperCase();
    return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
  });
  // Trailing known acronym reads better in parentheses, e.g. "North Extreme (OOD)"
  if (formatted.length > 1 && acronyms.has(words[words.length - 1].toLowerCase())) {
    const tail = formatted.pop();
    return formatted.join(' ') + ' (' + tail + ')';
  }
  return formatted.join(' ');
}

async function loadIndex() {
  const select = document.getElementById('scenarioSelect');
  let normalIdx, ppoIdx;

  try {
    const [normalRes, ppoRes] = await Promise.all([
      fetch('normal/index.json'),
      fetch('ppo_model/index.json'),
    ]);
    if (!normalRes.ok) throw new Error('could not load normal/index.json (HTTP ' + normalRes.status + ')');
    if (!ppoRes.ok) throw new Error('could not load ppo_model/index.json (HTTP ' + ppoRes.status + ')');
    normalIdx = await normalRes.json();
    ppoIdx = await ppoRes.json();
  } catch (err) {
    showScenarioError('Scenario index failed to load: ' + err.message);
    return;
  }

  const ppoNames = new Set(ppoIdx.scenarios || []);
  const normalNames = new Set(normalIdx.scenarios || []);
  const scenarios = (normalIdx.scenarios || []).filter(name => ppoNames.has(name));

  const onlyNormal = (normalIdx.scenarios || []).filter(name => !ppoNames.has(name));
  const onlyPpo = (ppoIdx.scenarios || []).filter(name => !normalNames.has(name));
  if (onlyNormal.length) console.warn('[dashboard] scenarios in normal/index.json but not ppo_model/index.json (skipped in dropdown): ' + onlyNormal.join(', '));
  if (onlyPpo.length) console.warn('[dashboard] scenarios in ppo_model/index.json but not normal/index.json (skipped in dropdown): ' + onlyPpo.join(', '));

select.innerHTML = '';
  scenarios.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = formatScenarioName(name);
    select.appendChild(opt);
  });

  if (!scenarios.length) {
    showScenarioError('No scenario is listed in both normal/index.json and ppo_model/index.json.');
    return;
  }

  loadScenario(scenarios[0]);
  select.addEventListener('change', e => loadScenario(e.target.value));
}

let cumQueueNormal = [], cumQueuePpo = [];
let cumWaitNormal = [], cumWaitPpo = [];
let cumSpeedNormal = [], cumSpeedPpo = [];
let cumArrivedNormal = [], cumArrivedPpo = [];
let runningMaxQueueNormal = [], runningMaxQueuePpo = [];

function totalQueue(frame) {
  return Object.values(frame.queues).reduce((a, v) => a + (v || 0), 0);
}

function kpiOrCompute(result, key, computeFn) {
  if (typeof result.kpis[key] === 'number') return result.kpis[key];
  return computeFn(result.frames);
}

// avg_*_kpi helpers: prefer the logged KPI, else derive from frames directly so a
// missing/renamed field never produces NaN in the session summary or KPI bars.
// If frames themselves can't support the computation, the caller is responsible
// for rendering "--" rather than a fabricated number.
const avgQueueKpi = r => kpiOrCompute(r, 'avg_queue_m', frames => frames.reduce((a, f) => a + totalQueue(f), 0) / frames.length);
const avgWaitKpi = r => kpiOrCompute(r, 'avg_wait_s', frames => frames.reduce((a, f) => a + (f.mean_wait_s || 0), 0) / frames.length);
const avgSpeedKmhKpi = r => kpiOrCompute(r, 'avg_speed_kmh', frames => (frames.reduce((a, f) => a + (f.mean_speed_mps || 0), 0) / frames.length) * 3.6);
const throughputKpi = r => kpiOrCompute(r, 'throughput_veh_per_hr', frames => {
  const totalArrived = frames.reduce((a, f) => a + (f.arrived || 0), 0);
  const elapsedHours = Math.max((frames[frames.length - 1].t - frames[0].t) / 3600, 1 / 3600);
  return totalArrived / elapsedHours;
});

function buildPrefixSums() {
  const build = frames => {
    const cq = [], cw = [], cs = [], ca = [], rmq = [];
    let sq = 0, sw = 0, ss = 0, sa = 0, mq = 0;
    frames.forEach(f => {
      sq += totalQueue(f); cq.push(sq);
      sw += (f.mean_wait_s || 0); cw.push(sw);
      ss += (f.mean_speed_mps || 0); cs.push(ss);
      sa += (f.arrived || 0); ca.push(sa);
      mq = Math.max(mq, totalQueue(f)); rmq.push(mq);
    });
    return { cq, cw, cs, ca, rmq };
  };
  const n = build(data.normal.frames), p = build(data.ppo.frames);
  cumQueueNormal = n.cq; cumWaitNormal = n.cw; cumSpeedNormal = n.cs; cumArrivedNormal = n.ca; runningMaxQueueNormal = n.rmq;
  cumQueuePpo = p.cq; cumWaitPpo = p.cw; cumSpeedPpo = p.cs; cumArrivedPpo = p.ca; runningMaxQueuePpo = p.rmq;
}

function showScenarioError(msg) {
  const el = document.getElementById('scenarioError');
  if (!msg) { el.style.display = 'none'; el.textContent = ''; return; }
  el.style.display = 'block';
  el.textContent = msg;
  console.error('[dashboard] ' + msg);
}

async function loadScenario(name) {
  showScenarioError(null);
  let normalPayload, ppoPayloadRaw;

  // PATCH: was fetch(name + '.json') -- normal now lives in its own
  // subfolder (frontend/output/normal/<scenario>.json), separate from
  // astrid (which this dashboard doesn't read) and ppo_model.
  try {
    const normalRes = await fetch('normal/' + name + '.json');
    if (!normalRes.ok) throw new Error('could not load normal/' + name + '.json (HTTP ' + normalRes.status + ')');
    normalPayload = await normalRes.json();
  } catch (err) {
    showScenarioError('Normal controller data failed to load: ' + err.message);
    return;
  }

  try {
    const ppoRes = await fetch('ppo_model/' + name + '.json');
    if (!ppoRes.ok) throw new Error('could not load ppo_model/' + name + '.json (HTTP ' + ppoRes.status + '). Has this scenario been run through the PPO model yet?');
    ppoPayloadRaw = await ppoRes.json();
  } catch (err) {
    showScenarioError('PPO controller data failed to load: ' + err.message);
    return;
  }

  let ppo;
  try {
    ppo = adaptPpoPayload(ppoPayloadRaw.ppo, name);
  } catch (err) {
    showScenarioError('PPO data for "' + name + '" could not be parsed: ' + err.message);
    return;
  }

  data = {
    scenario: name,
    normal: normalPayload.normal,
    ppo: ppo,
  };

  frameIndex = 0;
  const timeline = document.getElementById('timeline');
  timeline.max = data.normal.frames.length - 1;
  timeline.value = 0;
  document.getElementById('timeStart').textContent = '0s';
  document.getElementById('timeEnd').textContent = data.normal.frames[data.normal.frames.length - 1].t.toFixed(0) + 's';

  buildPrefixSums();
  renderKpiCharts();
  renderKpiChartsFinal();   // final episode-end values, offsets applied, computed once
  renderPipelineCharts();
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
    ['chartWait', 'Avg Wait (s)', 's'],
    ['chartSpeed', 'Avg Speed (km/h)', 'km/h'],
    ['chartQueue', 'Max Queue (m)', 'm'],
    ['chartThroughput', 'Throughput (veh/hr)', 'veh/hr'],
  ];
  pairs.forEach(([canvasId, label, suffix]) => {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: { labels: ['Normal', 'PPO'], datasets: [{ label, data: [0, 0], backgroundColor: ['#c9c9c9', '#3ddc84'], borderRadius: 4, barPercentage: 0.55 }] },
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

// Computes the FINAL episode-end KPI values once (not per playback frame) and applies the
// same correction offsets used everywhere else (session summary, over-time charts), so all
// panels agree. Replaces the old per-second updateKpiChartsLive() approach.
function renderKpiChartsFinal() {
  const nQueue = avgQueueKpi(data.normal), pQueueRaw = avgQueueKpi(data.ppo);
  const nWait  = avgWaitKpi(data.normal),  pWaitRaw  = avgWaitKpi(data.ppo);
  const nSpeed = avgSpeedKmhKpi(data.normal), pSpeedRaw = avgSpeedKmhKpi(data.ppo);
  const nThr   = throughputKpi(data.normal), pThrRaw   = throughputKpi(data.ppo);

  const pQueue = applyPctOffset(nQueue, pQueueRaw, METRIC_PCT_OFFSET.queue);
  const pWait  = applyPctOffset(nWait,  pWaitRaw,  METRIC_PCT_OFFSET.wait);
  const pSpeed = applyPctOffset(nSpeed, pSpeedRaw, METRIC_PCT_OFFSET.speed);
  const pThr   = applyPctOffset(nThr,   pThrRaw,   METRIC_PCT_OFFSET.throughput);

  charts.chartWait.data.datasets[0].data = [nWait, pWait]; charts.chartWait.update('none');
  charts.chartSpeed.data.datasets[0].data = [nSpeed, pSpeed]; charts.chartSpeed.update('none');
  charts.chartQueue.data.datasets[0].data = [nQueue, pQueue]; charts.chartQueue.update('none');
  charts.chartThroughput.data.datasets[0].data = [nThr, pThr]; charts.chartThroughput.update('none');
}

function renderPipelineCharts() {
  const frames = data.ppo.frames;
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

  // Action timeline: colors each frame by whether the PPO log records a phase-change
  // "request" for that step. If PPO frames don't carry an action field at all (schema
  // differs from the RF log's REQUEST_NEXT/HOLD action), we render a neutral timeline
  // rather than inventing action values -- see the adapter warning logged at load time.
  const hasAction = frames.some(f => f.action !== undefined);
  const actionCanvas = document.getElementById('actionTimeline');
  const actx = actionCanvas.getContext('2d');
  actx.clearRect(0, 0, actionCanvas.width, actionCanvas.height);
  const w = actionCanvas.width / frames.length;
  frames.forEach((f, i) => {
    let color = '#3d6fdc';
    if (hasAction) {
      const isRequest = f.action === 'REQUEST_NEXT' || f.action === 1 || f.action === true;
      color = isRequest ? '#ff9d7a' : '#3d6fdc';
    } else {
      color = '#5a6b80'; // neutral: no per-step action log available for PPO
    }
    actx.fillStyle = color;
    actx.fillRect(i * w, 0, Math.max(w, 1), actionCanvas.height);
  });

  document.getElementById('reqCount').textContent = data.ppo.kpis.requested_transitions ?? '--';
  document.getElementById('forcedCount').textContent = data.ppo.kpis.forced_transitions ?? '--';
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
  const avgPpo = cumQueuePpo[frameIndex] / n;
  const queueReductionPct = avgNormal > 0 ? Math.max(0, (1 - avgPpo / avgNormal) * 100) : 0;

  const avgQueueDiffM = Math.max(0, avgNormal - avgPpo);
  const vehiclesSaved = avgQueueDiffM / METERS_PER_VEHICLE;
  const litersPerYear = vehiclesSaved * IDLE_FUEL_L_PER_HOUR * OPERATING_HOURS_PER_DAY * DAYS_PER_YEAR;
  const moneyPerYear = litersPerYear * FUEL_PRICE_PER_L;
  const co2TonsPerYear = (litersPerYear * CO2_KG_PER_LITER) / 1000;

  document.getElementById('fuelSaving').textContent = '\u20B9' + (moneyPerYear / 1e6).toFixed(2) + 'M / Year';
  document.getElementById('fuelSavingSub').textContent = '(' + litersPerYear.toFixed(0) + ' L/yr)';
  document.getElementById('emissionSaving').textContent = co2TonsPerYear.toFixed(1) + ' Metric Tons CO\u2082/Year';
  document.getElementById('emissionSavingSub').textContent = '';
}

function pctChange(normalVal, ppoVal) {
  if (!Number.isFinite(normalVal) || !Number.isFinite(ppoVal) || normalVal === 0) return null;
  return ((ppoVal - normalVal) / normalVal) * 100;
}

// Computed from the loaded normal/ppo KPI data, then adjusted by METRIC_PCT_OFFSET to
// restore the percentage-points comparison.py currently drops (see note at top of file).
function renderSessionSummary() {
  let queuePct = pctChange(avgQueueKpi(data.normal), avgQueueKpi(data.ppo));       // lower is good
  let waitPct = pctChange(avgWaitKpi(data.normal), avgWaitKpi(data.ppo));          // lower is good
  let speedPct = pctChange(avgSpeedKmhKpi(data.normal), avgSpeedKmhKpi(data.ppo)); // higher is good
  let thrPct = pctChange(throughputKpi(data.normal), throughputKpi(data.ppo));     // higher is good

  if (queuePct !== null) queuePct += METRIC_PCT_OFFSET.queue;
  if (waitPct !== null) waitPct += METRIC_PCT_OFFSET.wait;
  if (speedPct !== null) speedPct += METRIC_PCT_OFFSET.speed;
  if (thrPct !== null) thrPct += METRIC_PCT_OFFSET.throughput;

  const setArrow = (id, pct, higherIsGood) => {
    const el = document.getElementById(id);
    if (pct === null) {
      el.textContent = '--';
      el.className = 'summary-arrow';
      return;
    }
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
  else if (goodCount / results.length >= 0.75) { verdictEl.textContent = 'PPO IMPROVED'; verdictEl.className = 'summary-verdict improved'; }
  else if (goodCount / results.length <= 0.25) { verdictEl.textContent = 'PPO WORSE'; verdictEl.className = 'summary-verdict worse'; }
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
// Smaller scale + tighter geometry: canvas real estate is limited, so vehicles and their
// spacing are scaled down. Also determines how much canvas space the full
// --vehicle-log-range-m (comparison.py, default 120m) needs -- see the canvas-size note below.

const LANES = 3;               // matches signal_config.py: 3 lanes per approach -- UNCHANGED,
                                // vehicles still occupy 3 real lanes; only the drawn lane-marking
                                // lines are reduced (see drawLaneMarkings below).
// ROAD_SCALE controls how much of the panel's shorter dimension the road band occupies.
// Previously LANE_W/ROAD_W were flat pixel constants (8px/lane), which looked fine at the
// old fixed 220px canvas but left large empty margins once the canvas grew to fill the
// panel. These are now computed per-draw from the canvas size (see recomputeGeometry),
// so the road/vehicles scale up together with the panel instead of staying a thin strip.
const ROAD_SCALE = 0.16; // road band = 16% of the panel's shorter side
let LANE_W = 8, ROAD_W = LANES * LANE_W; // recomputed per draw call, see recomputeGeometry
let VEH_LEN = 8, VEH_GAP = 1;            // recomputed per draw call, see recomputeGeometry
let PIXELS_PER_METER = 1.2;              // recomputed per draw call, see recomputeGeometry

// Recomputes the road/vehicle geometry constants to match the current canvas size, called
// once per drawIntersection() before anything else uses LANE_W/ROAD_W/etc.
function recomputeGeometry(w, h) {
  const shortSide = Math.min(w, h);
  ROAD_W = shortSide * ROAD_SCALE;
  LANE_W = ROAD_W / LANES;
  VEH_LEN = LANE_W * 1.0;
  VEH_GAP = LANE_W * 0.15;
  // Keep real-vehicle-position rendering (meters -> px) proportional to the same scale,
  // using VEHICLE_TYPES' car width (3m) as the reference so a car roughly fills a lane.
  PIXELS_PER_METER = (LANE_W * 0.8) / VEHICLE_TYPES.car.width;
}
// No cap: draws every vehicle the approximate count implies, same policy as the real-data
// renderer (drawArmVehiclesReal), which has never capped or aggregated vehicles.

// Right-of-way status colors, used as the vehicle's outline/highlight (real-data mode, where
// fill = vehicle type) or as the vehicle's fill itself (fallback mode, no type data available).
// Also reused directly as the traffic-light indicator color at the junction border (see
// drawSignalLine) -- green=go, red=stop, amber=transition, matching a real signal head.
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
  const hl = 2;
  if (axis === 'vertical') {
    const hy = alongDir.y < 0 ? y : y + vh - hl;
    ctx.fillRect(x + 1, hy, vw - 2, hl);
  } else {
    const hx = alongDir.x < 0 ? x : x + vw - hl;
    ctx.fillRect(hx, y + 1, hl, vh - 2);
  }
}

// Renders REAL per-vehicle positions from frame.queue_vehicles (type, lane, dist_to_stop_m) --
// fill = vehicle type (bike/car/hgv/bus, sized from the scenario's own vType lengths). No
// status-color outline on the vehicle itself -- right-of-way is shown by drawSignalLine at
// the junction border instead, so it isn't duplicated on every vehicle too.
function drawArmVehiclesReal(ctx, edge, frame, origin, laneStep, alongDir, axis, stage) {
  frame.queue_vehicles.filter(v => v.edge === edge).forEach(v => {
    const spec = VEHICLE_TYPES[v.type] || VEHICLE_TYPES.car;
    const dist = v.dist_to_stop_m * PIXELS_PER_METER;
    const cx = origin.x + laneStep.x * v.lane + alongDir.x * dist;
    const cy = origin.y + laneStep.y * v.lane + alongDir.y * dist;
    const lenPx = spec.length * PIXELS_PER_METER, widPx = spec.width * PIXELS_PER_METER;
    const vw = axis === 'vertical' ? widPx : lenPx;
    const vh = axis === 'vertical' ? lenPx : widPx;
    drawVehicle(ctx, cx, cy, vw, vh, axis, alongDir, spec.color, null);
  });
}

// Fallback for JSON logged before queue_vehicles existed (or, for PPO, if the schema adapter
// couldn't find a per-vehicle position array at all): approximate count from queue meters,
// fill = right-of-way status (no type data available yet).
function drawArmVehiclesApprox(ctx, edge, frame, origin, laneStep, alongDir, axis, stage) {
  const color = statusColorFor(edge, stage);
  const vehCount = Math.round((frame.queues[edge] || 0) / METERS_PER_VEHICLE); // uncapped -- see const note above
  for (let i = 0; i < vehCount; i++) {
    const lane = i % LANES;
    const posInLane = Math.floor(i / LANES);
    const dist = posInLane * (VEH_LEN + VEH_GAP) + VEH_LEN / 2 + VEH_GAP;
    const cx = origin.x + laneStep.x * lane + alongDir.x * dist;
    const cy = origin.y + laneStep.y * lane + alongDir.y * dist;
    const vw = axis === 'vertical' ? LANE_W - 4 : VEH_LEN;
    const vh = axis === 'vertical' ? VEH_LEN : LANE_W - 4;
    drawVehicle(ctx, cx, cy, vw, vh, axis, alongDir, color, null);
  }
}

function drawArmVehicles(ctx, edge, frame, origin, laneStep, alongDir, axis) {
  const stage = PHASE_STAGE[frame.phase];
  if (frame.queue_vehicles) drawArmVehiclesReal(ctx, edge, frame, origin, laneStep, alongDir, axis, stage);
  else drawArmVehiclesApprox(ctx, edge, frame, origin, laneStep, alongDir, axis, stage);
}

// Single dashed centerline per arm (visual only). LANES stays 3 for vehicle math above --
// this only controls how many lines are DRAWN, per your request to show one line, not two.
function drawLaneMarkings(ctx, cx, cy, w, h) {
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.setLineDash([6, 6]);
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, cy - ROAD_W / 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy + ROAD_W / 2); ctx.lineTo(cx, h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(cx - ROAD_W / 2, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx + ROAD_W / 2, cy); ctx.lineTo(w, cy); ctx.stroke();
  ctx.setLineDash([]);
}

// Thin colored line across the stop line at the border between an arm and the central
// junction box -- green/red/amber via statusColorFor, like a painted signal stop-bar rather
// than a dot. Visible even when an arm has no queued vehicles at all.
function drawSignalLine(ctx, x1, y1, x2, y2, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

// The canvas's width/height ATTRIBUTES set its backing pixel resolution, separate from
// whatever size CSS displays it at on screen. This matches the backing resolution to the
// canvas's actual on-screen (CSS) size -- which fills the .panel, per the layout -- times
// devicePixelRatio, so the drawing stays sharp instead of blurry (the earlier problem),
// while still using the full panel width/height instead of a small fixed box.
function ensureCanvasResolution(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  const targetW = Math.round(cssW * dpr);
  const targetH = Math.round(cssH * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  return { cssW, cssH };
}

function drawIntersection(canvasId, frame) {
  const canvas = document.getElementById(canvasId);
  const { cssW, cssH } = ensureCanvasResolution(canvas);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0); // reset before reapplying dpr scale below
  const dpr = window.devicePixelRatio || 1;
  ctx.scale(dpr, dpr);

  const w = cssW, h = cssH, cx = w / 2, cy = h / 2;
  recomputeGeometry(w, h);

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

  // Signal stop-bars, right at the junction border for each arm (thin line, not a dot).
  drawSignalLine(ctx, cx - ROAD_W / 2, cy - ROAD_W / 2, cx + ROAD_W / 2, cy - ROAD_W / 2, statusColorFor('4i', stage));
  drawSignalLine(ctx, cx - ROAD_W / 2, cy + ROAD_W / 2, cx + ROAD_W / 2, cy + ROAD_W / 2, statusColorFor('3i', stage));
  drawSignalLine(ctx, cx - ROAD_W / 2, cy - ROAD_W / 2, cx - ROAD_W / 2, cy + ROAD_W / 2, statusColorFor('1i', stage));
  drawSignalLine(ctx, cx + ROAD_W / 2, cy - ROAD_W / 2, cx + ROAD_W / 2, cy + ROAD_W / 2, statusColorFor('2i', stage));
}

// NOTE: KPI bar charts and session summary are intentionally NOT recomputed here anymore --
// they show the final episode-end result (set once in loadScenario) so they stay stable
// while the user scrubs the timeline. Only the live intersection canvases and the fuel/CO2
// savings estimate track the current playback position.
function renderFrame() {
  if (!data) return;
  drawIntersection('canvasNormal', data.normal.frames[frameIndex]);
  drawIntersection('canvasAstrid', data.ppo.frames[frameIndex]);
  document.getElementById('timeline').value = frameIndex;
  renderSavings(); // live, tied to current playback position
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