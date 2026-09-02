/* =========================================================================
   ASTRID DASHBOARD — DATA LAYER
   -------------------------------------------------------------------------
   Two independent data sources feed this dashboard:

     1. NORMAL CONTROLLER  — you already have normal_controller.py running
        Webster timing against SUMO. Until it's wired up over a websocket/
        API, this file simulates plausible telemetry in the same shape so
        the layout and charts can be built and reviewed now.

     2. ASTRID CONTROLLER  — does not exist yet. The dashboard defaults to
        a "not connected" state for this side: no numbers are invented,
        every ASTRID field renders as "Not available" / "Waiting for
        ASTRID", and the panel is visually dimmed. Nothing about this
        side is guessed — see connectAstrid() below for how to activate it.

   WIRING UP THE REAL BACKEND
   -------------------------------------------------------------------------
   Replace the setInterval(...) tick at the bottom with something like:

     const normalWs = new WebSocket("ws://localhost:8000/normal-state");
     normalWs.onmessage = (evt) => { latestNormal = JSON.parse(evt.data); render(); };

     // once astrid_controller.py exists and is exposed similarly:
     const astridWs = new WebSocket("ws://localhost:8000/astrid-state");
     astridWs.onmessage = (evt) => {
       latestAstrid = JSON.parse(evt.data);
       ASTRID_CONNECTED = true;
       render();
     };

   As long as each message matches the field names used below (queue_m,
   delay_s, throughput_veh_h, phase, phase_is_green, phase_elapsed_s,
   action, confidence, reason, estimated_queue_m, predicted_queue_m,
   queue_growth, upstream_traffic), no other code needs to change.
   ========================================================================= */

// Flip this to true once astrid_controller.py is producing real state and
// wired to a websocket/API. Until then the ASTRID panel stays in the
// "not connected" state rather than showing fabricated numbers.
let ASTRID_CONNECTED = false;

const HISTORY_LEN = 40;
let history = { normal: { queue: [], delay: [], throughput: [] }, astrid: { queue: [], delay: [], throughput: [] } };
let simTime = 0;

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

/* -------- stand-in for real normal_controller.py / state_builder.py telemetry -------- */
function generateMockNormalState(){
  const cyclePos = (simTime % 60);
  const demandWave = 0.5 + 0.5 * Math.sin(simTime / 45);
  const phaseGreen = cyclePos < 30;
  const queue = clamp(60 + 90 * demandWave + (phaseGreen ? -20 : 25) + (Math.random()*10-5), 5, 260);
  const delay = clamp(queue * 0.32 + Math.random()*4, 3, 95);
  const throughput = clamp(950 - queue*1.1 + Math.random()*20, 300, 1000);
  const stops = Math.round(clamp(queue/12 + Math.random()*2, 0, 40));

  return {
    phase: phaseGreen ? "East-West Green" : "North-South Green",
    phase_is_green: phaseGreen,
    phase_elapsed_s: Math.round(phaseGreen ? cyclePos : cyclePos - 30),
    action: "Fixed timing (Webster cycle, no adaptation)",
    queue_m: queue,
    waiting_vehicles: Math.round(queue / 7),
    avg_speed_kmh: clamp(34 - queue*0.08, 3, 40),
    throughput_veh_h: throughput,
    delay_s: delay,
    stops: stops,
    demandWave // internal use only, not rendered
  };
}

/* -------- ASTRID: null/"waiting" shape until connectAstrid() is called -------- */
function pendingAstridState(){
  return {
    phase: null, phase_is_green: null, phase_elapsed_s: null,
    action: null, confidence: null, reason: null,
    queue_m: null, waiting_vehicles: null, avg_speed_kmh: null,
    throughput_veh_h: null, delay_s: null, stops: null,
    estimated_queue_m: null, predicted_queue_m: null,
    queue_growth: null, upstream_traffic: null
  };
}

// Optional: a mock ASTRID generator you can use to preview the "connected"
// look before the real controller exists. Call connectAstrid(true) in the
// console, or flip ASTRID_CONNECTED above, to try it.
function generateMockAstridState(normalState, demandWave){
  const cyclePos = ((simTime + 12) % 52);
  const phaseGreen = cyclePos < 32;
  const queue = clamp(normalState.queue_m * (0.55 + Math.random()*0.15), 3, 200);
  const delay = clamp(queue * 0.27 + Math.random()*3, 2, 80);
  const throughput = clamp(normalState.throughput_veh_h * (1.06 + Math.random()*0.08), 300, 1100);
  const stops = Math.round(clamp(queue/14 + Math.random()*1.5, 0, 35));
  const prevQueue = history.astrid.queue.at(-1);
  const growth = prevQueue != null && queue > prevQueue + 6 ? "High" : (prevQueue != null && queue > prevQueue ? "Moderate" : "Low");
  const predicted = clamp(queue + (growth === "High" ? 25 : growth === "Moderate" ? 10 : -8), 0, 260);

  return {
    phase: phaseGreen ? "East-West Green" : "North-South Green",
    phase_is_green: phaseGreen,
    phase_elapsed_s: Math.round(phaseGreen ? cyclePos : cyclePos - 32),
    action: growth === "High" ? "Extend" : "Maintain",
    confidence: clamp(0.7 + Math.random()*0.25, 0, 1),
    reason: growth === "High" ? "High queue growth detected on EW approach" : "Queue stable; holding current phase",
    queue_m: queue,
    waiting_vehicles: Math.round(queue / 7),
    avg_speed_kmh: clamp(34 - queue*0.06, 3, 42),
    throughput_veh_h: throughput,
    delay_s: delay,
    stops: stops,
    estimated_queue_m: queue * (0.97 + Math.random()*0.06),
    predicted_queue_m: predicted,
    queue_growth: growth,
    upstream_traffic: demandWave > 0.6 ? "High" : demandWave > 0.3 ? "Moderate" : "Low"
  };
}

// Call this once astrid_controller.py is ready and wired up (or pass true
// to preview with the mock generator above).
function connectAstrid(useMock = false){
  ASTRID_CONNECTED = true;
  window.__astridMock = useMock;
}
window.connectAstrid = connectAstrid; // exposed for console use while testing

/* ============================== RENDER ================================ */

function fmt(v, unit=""){ return (v === null || v === undefined || Number.isNaN(v)) ? null : `${Math.round(v*10)/10}${unit}`; }
function naSpan(){ return `<span class="na-note">Not available</span>`; }

function renderScenario(s){
  document.getElementById('scnName').textContent = s.name ?? '—';
  document.getElementById('scnTime').textContent = s.time_s != null ? `${s.time_s}s` : '—';
  document.getElementById('scnDemand').textContent = s.demand ?? '—';
  document.getElementById('scnWeather').textContent = s.weather ?? '—';
}

function renderControllerHeader(prefix, c){
  const lampEl = document.getElementById(prefix+'Lamp');
  if (c.phase_is_green === null || c.phase_is_green === undefined){
    lampEl.className = 'signal-lamp lamp-off';
  } else {
    lampEl.className = 'signal-lamp ' + (c.phase_is_green ? 'lamp-green' : 'lamp-red');
  }
  document.getElementById(prefix+'Phase').textContent = c.phase ?? '—';
  document.getElementById(prefix+'Elapsed').textContent = c.phase_elapsed_s != null ? `${c.phase_elapsed_s}s elapsed` : '—';
}

function kpiMini(label, value, unit=""){
  const v = fmt(value, unit);
  return `<div class="kpi-mini"><div class="val ${v===null?'na':''}">${v===null? 'Not available' : v}</div><div class="lbl">${label}</div></div>`;
}

function renderKpiMiniGrid(elId, c){
  document.getElementById(elId).innerHTML = [
    kpiMini('QUEUE', c.queue_m, ' m'),
    kpiMini('WAITING VEHICLES', c.waiting_vehicles),
    kpiMini('AVG SPEED', c.avg_speed_kmh, ' km/h'),
    kpiMini('THROUGHPUT', c.throughput_veh_h, ' v/h'),
    kpiMini('DELAY', c.delay_s, ' s'),
    kpiMini('STOPS', c.stops)
  ].join('');
}

function renderComparisonTable(n, a){
  const rows = [
    ['Queue length', n.queue_m, a.queue_m, ' m', true],
    ['Average delay', n.delay_s, a.delay_s, ' s', true],
    ['Throughput', n.throughput_veh_h, a.throughput_veh_h, ' veh/h', false],
    ['Average speed', n.avg_speed_kmh, a.avg_speed_kmh, ' km/h', false],
    ['Stops', n.stops, a.stops, '', true],
  ];
  const body = rows.map(([label, nv, av, unit, lowerIsBetter]) => {
    const nCell = fmt(nv, unit) ?? naSpan();
    const aCell = fmt(av, unit) ?? naSpan();
    if (nv != null && av != null && nv !== 0) {
      const pct = ((av - nv) / nv) * 100;
      const improved = lowerIsBetter ? pct < 0 : pct > 0;
      const cls = Math.abs(pct) < 1 ? 'diff-flat' : (improved ? 'diff-good' : 'diff-bad');
      const arrow = pct === 0 ? '' : (pct > 0 ? 'up' : 'down');
      return `<tr><td>${label}</td><td class="normal-col">${nCell}</td><td class="astrid-col">${aCell}</td><td class="${cls}"><span class="${arrow}">${Math.abs(pct).toFixed(0)}%</span></td></tr>`;
    }
    return `<tr><td>${label}</td><td class="normal-col">${nCell}</td><td class="astrid-col">${aCell}</td><td>${naSpan()}</td></tr>`;
  }).join('');
  document.getElementById('kpiTableBody').innerHTML = body;
}

function badgeFor(level){
  if (level == null) return naSpan();
  const cls = level === 'High' ? 'badge-high' : level === 'Moderate' ? 'badge-med' : 'badge-low';
  return `${level}<span class="badge ${cls}">&nbsp;</span>`;
}

function renderWhy(a){
  const rows = [
    ['Queue growth', badgeFor(a.queue_growth)],
    ['Upstream traffic', badgeFor(a.upstream_traffic)],
    ['Current phase', a.phase ?? naSpan()],
    ['Phase elapsed', a.phase_elapsed_s != null ? `${a.phase_elapsed_s}s` : naSpan()],
    ['Estimated queue', fmt(a.estimated_queue_m, ' m') ?? naSpan()],
    ['Predicted queue', fmt(a.predicted_queue_m, ' m') ?? naSpan()],
  ];
  document.getElementById('whyGrid').innerHTML = rows.map(([k,v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join('');
  const concEl = document.getElementById('whyConclusion');
  if (!ASTRID_CONNECTED){
    concEl.textContent = 'Waiting for ASTRID controller to be connected.';
    concEl.classList.add('pending');
  } else {
    concEl.classList.remove('pending');
    concEl.textContent = a.action
      ? `→ ASTRID recommends: ${a.action.toLowerCase()}${a.reason ? ' — ' + a.reason.toLowerCase() : ''}.`
      : 'Waiting for ASTRID';
  }
}

function renderAstridPanelState(){
  const panel = document.getElementById('astridPanel');
  const tag = document.getElementById('astridTag');
  panel.classList.toggle('pending', !ASTRID_CONNECTED);
  tag.classList.toggle('pending', !ASTRID_CONNECTED);
  tag.textContent = ASTRID_CONNECTED ? 'ADAPTIVE' : 'NOT CONNECTED';
}

/* -------- junction (schematic — dot count reflects queue estimate, not real vehicle positions) -------- */
function renderJunction(n, a){
  const svg = document.getElementById('junctionSvg');
  const cx = 310, cy = 180;
  // Use ASTRID's queue if connected, otherwise fall back to Normal's — never fabricate a number neither side reports.
  const queueForViz = a.queue_m != null ? a.queue_m : n.queue_m;
  const activeGreenIsEW = (ASTRID_CONNECTED ? a.phase : n.phase)?.includes('East-West') ?? false;
  const activeGreenIsNS = (ASTRID_CONNECTED ? a.phase : n.phase)?.includes('North-South') ?? false;

  const queueDots = (queueM, dir) => {
    if (queueM == null) return '';
    const count = clamp(Math.round(queueM / 18), 0, 10);
    let dots = '';
    for (let i=0;i<count;i++){
      const off = 26 + i*20;
      let x=cx, y=cy;
      if (dir==='N'){ x=cx-14; y=cy-off; }
      if (dir==='S'){ x=cx+14; y=cy+off; }
      if (dir==='E'){ x=cx+off; y=cy-14; }
      if (dir==='W'){ x=cx-off; y=cy+14; }
      dots += `<circle cx="${x}" cy="${y}" r="4.2" fill="#c9d6da" opacity="0.85"/>`;
    }
    return dots;
  };
  const lamp = (isGreen, x, y) => `<circle cx="${x}" cy="${y}" r="6" fill="${isGreen ? 'var(--green)' : 'var(--red)'}" style="filter:drop-shadow(0 0 5px ${isGreen? '#4fbf8b':'#e05d4a'})"/>`;

  svg.innerHTML = `
    <rect x="0" y="0" width="620" height="360" fill="none"/>
    <rect x="${cx-30}" y="0" width="60" height="360" fill="#161e22"/>
    <rect x="0" y="${cy-30}" width="620" height="60" fill="#161e22"/>
    <line x1="${cx}" y1="0" x2="${cx}" y2="360" stroke="#2a373e" stroke-width="1" stroke-dasharray="6 6"/>
    <line x1="0" y1="${cy}" x2="620" y2="${cy}" stroke="#2a373e" stroke-width="1" stroke-dasharray="6 6"/>

    <rect x="${cx-30}" y="${cy-34}" width="60" height="3" fill="#3a4850"/>
    <rect x="${cx-30}" y="${cy+31}" width="60" height="3" fill="#3a4850"/>
    <rect x="${cx-34}" y="${cy-30}" width="3" height="60" fill="#3a4850"/>
    <rect x="${cx+31}" y="${cy-30}" width="3" height="60" fill="#3a4850"/>

    ${lamp(activeGreenIsEW, cx-46, cy-46)}
    ${lamp(activeGreenIsNS, cx+46, cy-46)}

    ${queueDots(queueForViz, 'N')}
    ${queueDots(queueForViz, 'S')}
    ${queueDots(queueForViz, 'E')}
    ${queueDots(queueForViz, 'W')}

    <text x="${cx}" y="30" fill="#7d939c" font-family="var(--mono)" font-size="11" text-anchor="middle">NORTH</text>
    <text x="${cx}" y="345" fill="#7d939c" font-family="var(--mono)" font-size="11" text-anchor="middle">SOUTH</text>
    <text x="20" y="${cy-40}" fill="#7d939c" font-family="var(--mono)" font-size="11">WEST</text>
    <text x="560" y="${cy-40}" fill="#7d939c" font-family="var(--mono)" font-size="11">EAST</text>
    <text x="${cx}" y="${cy+90}" fill="#4b5c63" font-family="var(--mono)" font-size="10" text-anchor="middle">schematic — dot count reflects queue estimate, not tracked vehicle positions</text>
  `;
}

/* -------- line charts (pure SVG, no libs) -------- */
function renderLineChart(svgId, seriesA, seriesB){
  const svg = document.getElementById(svgId);
  const w = 300, h = 120, pad = 6;
  const cleanA = seriesA.filter(v => v != null);
  const cleanB = seriesB.filter(v => v != null);
  const all = cleanA.concat(cleanB);
  if (all.length === 0){ svg.innerHTML = ''; return; }
  const max = Math.max(...all, 1), min = Math.min(...all, 0);
  const range = (max - min) || 1;
  const toPoints = (arr) => {
    const pts = arr.map((v,i) => v == null ? null : [
      pad + (i/(HISTORY_LEN-1)) * (w-2*pad),
      h - pad - ((v-min)/range) * (h-2*pad)
    ]);
    return pts.filter(Boolean).map(p => p.join(',')).join(' ');
  };
  svg.innerHTML = `
    <polyline points="${toPoints(seriesA)}" fill="none" stroke="var(--normal)" stroke-width="2" opacity="0.9"/>
    <polyline points="${toPoints(seriesB)}" fill="none" stroke="var(--astrid)" stroke-width="2"/>
  `;
}

function renderCharts(){
  renderLineChart('chartQueue', history.normal.queue, history.astrid.queue);
  renderLineChart('chartDelay', history.normal.delay, history.astrid.delay);
  renderLineChart('chartThroughput', history.normal.throughput, history.astrid.throughput);
}

function renderSummary(){
  const validPairs = history.normal.queue.filter((v,i) => v != null && history.astrid.queue[i] != null).length;
  if (!ASTRID_CONNECTED || validPairs < 5){
    document.getElementById('summaryMetrics').innerHTML =
      `<span class="na-note">${!ASTRID_CONNECTED ? 'Waiting for ASTRID controller to be connected.' : 'Collecting data — insufficient samples for a session summary yet.'}</span>`;
    const verdictEl = document.getElementById('verdict');
    verdictEl.textContent = 'INSUFFICIENT DATA';
    verdictEl.classList.add('insufficient');
    return;
  }
  const avg = (arr) => { const c = arr.filter(v=>v!=null); return c.reduce((s,v)=>s+v,0)/c.length; };
  const nQ = avg(history.normal.queue), aQ = avg(history.astrid.queue);
  const nD = avg(history.normal.delay), aD = avg(history.astrid.delay);
  const nT = avg(history.normal.throughput), aT = avg(history.astrid.throughput);
  const pct = (nv, av) => nv ? ((av-nv)/nv*100) : 0;
  const qDiff = pct(nQ, aQ), dDiff = pct(nD, aD), tDiff = pct(nT, aT);

  const metric = (label, val, lowerIsBetter) => {
    const improved = lowerIsBetter ? val < 0 : val > 0;
    return `<div class="summary-metric"><div class="val ${val<0?'down':'up'}" style="color:${improved?'var(--green)':'var(--red)'}">${Math.abs(val).toFixed(0)}%</div><div class="lbl">${label}</div></div>`;
  };
  document.getElementById('summaryMetrics').innerHTML =
    metric('Avg Queue', qDiff, true) + metric('Avg Delay', dDiff, true) + metric('Throughput', tDiff, false);

  const verdictEl = document.getElementById('verdict');
  verdictEl.classList.remove('insufficient');
  verdictEl.textContent = (qDiff < 0 && dDiff < 0 && tDiff > 0) ? 'ASTRID OUTPERFORMS BASELINE' : 'MIXED RESULT — SEE KPIs ABOVE';
}

/* ============================== MAIN LOOP ============================== */

function pushHistory(normal, astrid){
  history.normal.queue.push(normal.queue_m);
  history.normal.delay.push(normal.delay_s);
  history.normal.throughput.push(normal.throughput_veh_h);
  history.astrid.queue.push(astrid.queue_m);
  history.astrid.delay.push(astrid.delay_s);
  history.astrid.throughput.push(astrid.throughput_veh_h);
  for (const side of ['normal','astrid']) for (const k of ['queue','delay','throughput']) if (history[side][k].length > HISTORY_LEN) history[side][k].shift();
}

function render(){
  const normal = generateMockNormalState();
  const astrid = ASTRID_CONNECTED
    ? (window.__astridMock ? generateMockAstridState(normal, normal.demandWave) : (window.__latestAstrid || pendingAstridState()))
    : pendingAstridState();

  const scenario = {
    name: "Heavy East-West Demand",
    time_s: simTime,
    demand: normal.demandWave > 0.66 ? "Heavy" : normal.demandWave > 0.33 ? "Moderate" : "Light",
    weather: null
  };

  document.getElementById('liveLabel').textContent = ASTRID_CONNECTED ? 'LIVE' : 'LIVE (NORMAL ONLY)';
  document.querySelector('.live-dot').classList.toggle('paused', false);

  renderScenario(scenario);
  renderAstridPanelState();
  renderControllerHeader('n', normal);
  renderControllerHeader('a', astrid);
  renderKpiMiniGrid('nKpis', normal);
  renderKpiMiniGrid('aKpis', astrid);
  document.getElementById('nAction').textContent = normal.action ?? '—';
  document.getElementById('aAction').textContent = ASTRID_CONNECTED ? (astrid.action ?? '—') : 'Waiting for ASTRID';
  document.getElementById('aConf').textContent = astrid.confidence != null ? astrid.confidence.toFixed(2) : '—';
  document.getElementById('aReason').textContent = astrid.reason ?? '—';
  renderComparisonTable(normal, astrid);
  renderWhy(astrid);
  renderJunction(normal, astrid);
  pushHistory(normal, astrid);
  renderCharts();
  renderSummary();

  simTime += 5;
}

render();
setInterval(render, 2000);