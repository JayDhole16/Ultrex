"use strict";
// The dashboard reads two logs and never writes anything:
//   /api/live    live.jsonl, the move-by-move stream the arena animates
//   /api/events  events.jsonl, one line per finished generation, behind the charts and the feed
// Everything below is view state. The controls change what is drawn, never the run.

const LIVE_POLL_MS = 400;
const EVENTS_POLL_MS = 1000;
const CATCHUP_AT = 25;      // queued events before playback hurries to catch up
const CATCHUP_MS = 25;      // per-event time while hurrying
const DURATIONS = {         // how long each kind of event holds the stage, in ms at 1x
  run_start: 0, round_start: 120, negotiation_start: 320, decision: 320,
  deal: 900, no_deal: 700, round_end: 260, generation_end: 0, run_end: 0,
};
const W = 1000, H = 560;
const RING = { cx: 500, cy: 332, rx: 405, ry: 172 };  // sits low, leaving the top clear for the duel
const STAGE = { left: { x: 322, y: 300 }, right: { x: 678, y: 300 } };
const INPUT_NAMES = ["balance", "need (valuation)", "offered to me", "offered to them", "round", "rounds left", "my turn"];
const OUTCOMES = { agreement: ["✓", "agreement"], disagreement: ["✕", "no deal"], timeout: ["⏱", "timeout"] };
const MODE_STATUS = { live: ["●", "Live"], playing: ["▶", "Playing"], paused: ["❚❚", "Paused"] };

const $ = (id) => document.getElementById(id);
// While the page is working through a backlog, skip the flourishes: they are for moves you are watching.
const catchingUp = () => live.queue.length > CATCHUP_AT;
const num = (value) => (typeof value === "number" && Number.isFinite(value) ? value : null);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;  // log text is untrusted: never innerHTML
  return node;
}

function format(value, digits) {
  const v = num(value);
  return v === null ? "—" : v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function percent(value, digits = 1) {
  const v = num(value);
  return v === null ? "—" : `${(v * 100).toFixed(digits)}%`;
}

function ago(time) {
  const seconds = Math.max(0, Math.round((Date.now() - time) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

// =====================================================================================================
// Live stream
// =====================================================================================================

const live = {
  offset: 0, queue: [], playing: true, speed: 1, nextAt: 0,
  connection: "connecting", waiting: true, seen: 0,
};

const arena = {
  agents: new Map(),   // id -> view model, including its animated position and radius
  pairs: [], outcomes: new Map(), spotlight: null, duel: null,
  coins: [], floats: [], maxTokens: 20, round: 0, generation: 0, deals: 0, noDeals: 0, vocab: 0,
};

const net = { agent: null, action: null, activations: null, accept: null, beta: null, messages: null, value: null };

async function pollLive() {
  let delay = LIVE_POLL_MS;
  try {
    const response = await fetch(`/api/live?offset=${live.offset}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.reset) resetLive();
    const firstBatch = live.seen === 0;
    live.offset = data.offset;
    live.waiting = data.waiting;
    for (const event of data.events) live.queue.push(event);
    live.seen += data.events.length;
    if (firstBatch && live.queue.length > 30) {
      // Opening the page part-way through a run: fast-forward to what is happening now.
      const now = performance.now();
      while (live.queue.length > 12) applyLive(live.queue.shift(), now);
      live.nextAt = now;
    }
    live.connection = "ok";
    if (data.more) delay = 0;
  } catch (error) {
    live.connection = "error";
    delay = LIVE_POLL_MS * 3;
  }
  setTimeout(pollLive, delay);
}

function resetLive() {
  live.queue = [];
  live.seen = 0;
  arena.agents.clear();
  arena.pairs = [];
  arena.duel = null;
  arena.spotlight = null;
  arena.coins = [];
  arena.floats = [];
  arena.deals = arena.noDeals = 0;
}

function pump(now) {
  if (!live.playing) return;
  while (live.queue.length && now >= live.nextAt) {
    const event = live.queue.shift();
    applyLive(event, now);
    const hurry = live.queue.length > CATCHUP_AT;
    const duration = (DURATIONS[event.type] ?? 200) / live.speed;
    live.nextAt = now + (hurry ? Math.min(duration, CATCHUP_MS) : duration);
  }
}

function applyLive(event, now) {
  switch (event.type) {
    case "run_start":
      arena.vocab = event.config ? event.config.message_vocab || 0 : 0;
      buildNetworkPanel();
      break;
    case "round_start": {
      arena.round = event.round;
      const seen = new Set();
      for (const record of event.agents) {
        seen.add(record.id);
        upsertAgent(record);
      }
      for (const [id, agent] of arena.agents) if (!seen.has(id) && !agent.removeAt) arena.agents.delete(id);
      arena.pairs = event.pairs || [];
      arena.outcomes = new Map();
      arena.spotlight = event.spotlight;
      layout(now);
      break;
    }
    case "negotiation_start":
      arena.duel = {
        pair: event.pair, needs: event.needs, compute: event.compute, tokens: event.tokens,
        buyer: event.buyer, surplus: event.surplus, moved: event.compute_moved,
        moves: [], bubbles: new Map(), table: null, outcome: null, since: now,
      };
      break;
    case "decision":
      applyDecision(event, now);
      break;
    case "deal":
    case "no_deal":
      if (arena.duel) {
        arena.duel.outcome = event.type === "deal" ? "deal" : "no deal";
        arena.duel.since = now;
        arena.duel.result = event;
        if (event.type === "deal") {
          arena.duel.table = { keeps: event.split[0], gives: event.split[1], from: event.pair[0], settled: true };
          if (!catchingUp()) spawnCoins(event, now);
        }
      }
      if (event.type === "deal") arena.deals += 1; else arena.noDeals += 1;
      break;
    case "round_end":
      for (const balance of event.balances) upsertAgent(balance);
      for (const birth of event.births) {
        const agent = arena.agents.get(birth.id);
        if (!agent) continue;
        agent.parent = birth.parent;
        if (!catchingUp()) { agent.state = "newborn"; agent.stateUntil = now + 1600; }
        const parent = arena.agents.get(birth.parent);
        if (agent.x === null && parent && parent.x !== null) { agent.x = parent.x; agent.y = parent.y; }  // born from its parent
      }
      for (const id of event.deaths) {
        const agent = arena.agents.get(id);
        if (!agent) continue;
        if (catchingUp()) arena.agents.delete(id);
        else { agent.state = "dying"; agent.removeAt = now + 900; }
      }
      for (const trade of event.trades) arena.outcomes.set(trade.pair.join("-"), trade.outcome);
      layout(now);
      break;
    case "generation_end":
      arena.generation = event.generation;
      break;
  }
  renderTiles(historyEvent());
}

function upsertAgent(record) {
  let agent = arena.agents.get(record.id);
  if (!agent) {
    // x stays null until layout puts it somewhere: on the ring, or on top of its parent if it was just born.
    agent = { id: record.id, tokens: record.tokens, compute: record.compute || 0, x: null, y: null, r: 0, state: "", born: performance.now() };
    arena.agents.set(record.id, agent);
  }
  if (record.tokens !== undefined) agent.tokens = record.tokens;
  if (record.compute !== undefined) agent.compute = record.compute;
  if (record.lineage !== undefined) agent.lineage = record.lineage;
  arena.maxTokens = Math.max(arena.maxTokens, agent.tokens || 0);
}

function applyDecision(event, now) {
  const duel = arena.duel;
  if (duel) {
    duel.moves.push(event);
    const bubble = { until: now + 2600 };
    if (event.action === "talk") bubble.text = `m${event.message}`;
    else if (event.action === "offer") bubble.text = `${Math.round((1 - event.to_partner) * 100)} / ${Math.round(event.to_partner * 100)}`;
    else if (event.action === "accept") bubble.text = "deal";
    else bubble.text = "no";
    bubble.kind = event.action;
    duel.bubbles.set(event.agent, bubble);
    if (event.action === "offer") {
      duel.table = { keeps: event.tokens[0], gives: event.tokens[1], from: event.agent, since: now, settled: false };
    }
  }
  net.agent = event.agent;
  net.action = event.action;
  net.round = event.negotiation_round;
  net.activations = event.activations;
  net.accept = event.accept_prob;
  net.beta = event.beta;
  net.messages = event.message_probs || null;
  net.chosen = event.action === "talk" ? event.message : null;
  net.value = event.value;
  renderNetwork();
}

function spawnCoins(event, now) {
  const [a, b] = event.pair;
  const buyer = arena.duel && arena.duel.buyer === a ? a : b;
  const seller = buyer === a ? b : a;
  for (let i = 0; i < 7; i += 1) arena.coins.push({ from: buyer, to: seller, start: now + i * 70, duration: 620 });
  arena.floats.push({ id: seller, text: `+${format(event.price, 2)} tokens`, start: now, duration: 1800 });
  if (arena.duel) arena.floats.push({ id: buyer, text: `+${format(arena.duel.moved, 1)} compute`, start: now + 150, duration: 1800 });
  if (arena.coins.length > 40) arena.coins.splice(0, arena.coins.length - 40);
  if (arena.floats.length > 6) arena.floats.splice(0, arena.floats.length - 6);
}

// =====================================================================================================
// Arena drawing
// =====================================================================================================

function layout(now) {
  const ids = [...arena.agents.keys()].sort((a, b) => a - b);
  ids.forEach((id, index) => {
    const agent = arena.agents.get(id);
    const angle = (index / Math.max(ids.length, 1)) * Math.PI * 2 - Math.PI / 2;
    agent.ringX = RING.cx + Math.cos(angle) * RING.rx;
    agent.ringY = RING.cy + Math.sin(angle) * RING.ry;
    if (agent.x === null) { agent.x = agent.ringX; agent.y = agent.ringY; }  // no fly-in from nowhere
  });
}

function radiusFor(tokens) {
  return 7 + 15 * Math.sqrt(clamp(tokens, 0, arena.maxTokens) / Math.max(arena.maxTokens, 1));
}

function drawArena(now) {
  const svg = $("arena");
  const spot = arena.spotlight || [];
  const parts = [];

  for (const agent of arena.agents.values()) {
    const seat = spot.indexOf(agent.id);
    const target = seat === 0 ? STAGE.left : seat === 1 ? STAGE.right : { x: agent.ringX ?? RING.cx, y: agent.ringY ?? RING.cy };
    const targetR = seat >= 0 ? 30 : radiusFor(agent.tokens);
    const ease = catchingUp() ? 1 : 0.12;  // no drifting into place while hurrying through a backlog
    if (agent.x === null) { agent.x = target.x; agent.y = target.y; }
    agent.x += (target.x - agent.x) * ease;
    agent.y += (target.y - agent.y) * ease;
    agent.r += (targetR - agent.r) * (catchingUp() ? 1 : 0.15);
    if (agent.removeAt && now > agent.removeAt) arena.agents.delete(agent.id);
    if (agent.state && agent.stateUntil && now > agent.stateUntil) agent.state = "";
  }

  // Pair links, flashing green or red once the round's outcomes are known.
  for (const [a, b] of arena.pairs) {
    const from = arena.agents.get(a), to = arena.agents.get(b);
    if (!from || !to) continue;
    const outcome = arena.outcomes.get(`${a}-${b}`);
    const kind = outcome === "agreement" ? " deal" : outcome ? " nodeal" : "";
    parts.push(`<line class="link${kind}" x1="${from.x.toFixed(1)}" y1="${from.y.toFixed(1)}" x2="${to.x.toFixed(1)}" y2="${to.y.toFixed(1)}"/>`);
  }

  for (const agent of arena.agents.values()) {
    const seat = spot.indexOf(agent.id);
    const classes = ["agent", agent.state, seat === 0 ? "spot-a" : seat === 1 ? "spot-b" : ""].filter(Boolean).join(" ");
    const fade = agent.state === "dying" ? Math.max(0, (agent.removeAt - now) / 900) : 1;
    parts.push(`<g class="${classes}" opacity="${fade.toFixed(2)}">`);
    parts.push(`<circle cx="${agent.x.toFixed(1)}" cy="${agent.y.toFixed(1)}" r="${Math.max(agent.r, 0.1).toFixed(1)}"/>`);
    if (seat < 0) parts.push(`<text x="${agent.x.toFixed(1)}" y="${(agent.y + agent.r + 11).toFixed(1)}">${agent.id}</text>`);
    parts.push(`</g>`);
  }

  parts.push(drawDuel(now));
  for (const coin of arena.coins) {
    const progress = (now - coin.start) / coin.duration;
    if (progress < 0 || progress > 1) continue;
    const from = arena.agents.get(coin.from), to = arena.agents.get(coin.to);
    if (!from || !to) continue;
    const x = from.x + (to.x - from.x) * progress;
    const y = from.y + (to.y - from.y) * progress - Math.sin(progress * Math.PI) * 70;
    parts.push(`<circle class="coin" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"/>`);
  }
  arena.coins = arena.coins.filter((coin) => now < coin.start + coin.duration);

  for (const float of arena.floats) {
    const progress = (now - float.start) / float.duration;
    const agent = arena.agents.get(float.id);
    if (!agent || progress < 0 || progress > 1) continue;
    parts.push(`<text class="float-label" x="${agent.x.toFixed(1)}" y="${(agent.y - agent.r - 22 - progress * 34).toFixed(1)}" opacity="${(1 - progress).toFixed(2)}">${float.text}</text>`);
  }
  arena.floats = arena.floats.filter((float) => now < float.start + float.duration);

  parts.push(`<text class="stage-label" x="16" y="24">round ${arena.round} · generation ${arena.generation}</text>`);
  if (!arena.agents.size) {
    const message = live.waiting ? "This run has no live stream (start it with demo.py)" : "Waiting for the first round";
    parts.push(`<text class="stage-label" x="${W / 2}" y="${H / 2}" text-anchor="middle">${message}</text>`);
  }
  svg.innerHTML = parts.join("");
}

function bubble(x, y, text, kind) {
  const width = Math.max(46, text.length * 9 + 18);
  const accent = kind === "accept" ? ' fill-opacity="0.15"' : "";
  return (
    `<g><rect class="bubble" x="${(x - width / 2).toFixed(1)}" y="${(y - 30).toFixed(1)}" width="${width}" height="26" rx="13"${accent}/>` +
    `<text class="bubble-text" x="${x.toFixed(1)}" y="${(y - 12).toFixed(1)}">${text}</text></g>`
  );
}

function drawDuel(now) {
  const duel = arena.duel;
  if (!duel) return "";
  const [idA, idB] = duel.pair;
  const a = arena.agents.get(idA), b = arena.agents.get(idB);
  if (!a || !b) return "";
  const parts = [];
  const label = (agent, index) => {
    const role = duel.buyer === agent.id ? "buying compute" : "selling compute";
    parts.push(`<text class="duel-name" x="${agent.x.toFixed(1)}" y="${(agent.y + 52).toFixed(1)}">agent ${agent.id}</text>`);
    parts.push(`<text class="duel-stat" x="${agent.x.toFixed(1)}" y="${(agent.y + 68).toFixed(1)}">${role}</text>`);
    parts.push(`<text class="duel-stat" x="${agent.x.toFixed(1)}" y="${(agent.y + 83).toFixed(1)}">${format(agent.tokens, 1)} tokens · ${format(duel.compute[index], 1)} compute · need ${format(duel.needs[index], 2)}</text>`);
  };
  label(a, 0);
  label(b, 1);

  for (const [id, speech] of duel.bubbles) {
    if (now > speech.until) { duel.bubbles.delete(id); continue; }
    const agent = arena.agents.get(id);
    if (agent) parts.push(bubble(agent.x, agent.y - agent.r, speech.text, speech.kind));
  }

  if (duel.table) {
    const table = duel.table;
    const total = Math.max(1, table.keeps + table.gives);
    const width = 230, x = 500 - width / 2, y = 58;
    const proposer = arena.agents.get(table.from);
    const settle = table.settled ? 1 : clamp((now - (table.since || now)) / 320, 0, 1);
    const cx = proposer ? proposer.x + (500 - proposer.x) * settle : 500;
    parts.push(`<g opacity="${(0.35 + 0.65 * settle).toFixed(2)}" transform="translate(${(cx - 500).toFixed(1)} 0)">`);
    parts.push(`<rect class="card-offer" x="${x}" y="${y}" width="${width}" height="74" rx="10"/>`);
    parts.push(`<text class="offer-label" x="500" y="${y + 18}">on the table, agent ${table.from} proposes</text>`);
    parts.push(`<text class="offer-value" x="500" y="${y + 40}">${table.keeps} / ${table.gives}</text>`);
    const barWidth = width - 36;
    const keepWidth = (barWidth * table.keeps) / total;
    parts.push(`<rect class="offer-bar-keep" x="${x + 18}" y="${y + 50}" width="${keepWidth.toFixed(1)}" height="10" rx="2"/>`);
    parts.push(`<rect class="offer-bar-give" x="${(x + 20 + keepWidth).toFixed(1)}" y="${y + 50}" width="${Math.max(0, barWidth - keepWidth - 2).toFixed(1)}" height="10" rx="2"/>`);
    parts.push(`</g>`);
  }

  if (duel.outcome) {
    const deal = duel.outcome === "deal";
    const result = duel.result || {};
    const text = deal
      ? `✓ deal in ${result.rounds} round${result.rounds === 1 ? "" : "s"} · price ${format(result.price, 2)} tokens · surplus ${format(result.surplus, 2)}`
      : "✕ no deal — the compute stays where it was";
    parts.push(`<text class="banner ${deal ? "deal" : "nodeal"}" x="500" y="38">${text}</text>`);
  } else {
    parts.push(`<text class="duel-stat" x="500" y="38">negotiating over ${format(duel.surplus, 3)} tokens of gains from trade</text>`);
  }
  return parts.join("");
}

// =====================================================================================================
// Network panel
// =====================================================================================================

let panel = null;

function cellGrid(count) {
  const grid = el("div", "cells");
  const cells = [];
  for (let i = 0; i < count; i += 1) {
    const cell = el("div", "cell");
    cells.push(cell);
    grid.append(cell);
  }
  return { grid, cells };
}

function section(title, body, valueId) {
  const row = el("div", "net-row");
  const head = el("div", "head");
  head.append(el("span", "", title));
  const value = el("span", "value");
  if (valueId) value.id = valueId;
  head.append(value);
  row.append(head, body);
  return { row, value };
}

function buildNetworkPanel() {
  const inputs = el("div", "inputs");
  const layer1 = cellGrid(64);
  const layer2 = cellGrid(64);
  const accept = el("div", "prob-track");
  const acceptFill = el("div", "prob-fill");
  accept.append(acceptFill);
  const beta = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  beta.setAttribute("class", "beta");
  beta.setAttribute("viewBox", "0 0 240 70");
  const tokens = el("div", "tokens");

  const scale = el("div", "scale");
  scale.append(el("span", "", "−1"), el("div", "ramp"), el("span", "", "+1"));

  const inputSection = section("What it sees", inputs);
  const layer1Section = section("Hidden layer 1 · 64 tanh units", layer1.grid);
  const layer2Section = section("Hidden layer 2 · 64 tanh units", layer2.grid);
  const acceptSection = section("Accept probability", accept);
  const betaSection = section("Offer it would make (Beta)", beta);
  const tokenSection = section("Message probabilities", tokens);
  const valueSection = section("Value estimate", el("div", "caption", "what it expects this negotiation to be worth"));

  const container = $("net");
  container.replaceChildren(
    inputSection.row, layer1Section.row, layer2Section.row, scale,
    acceptSection.row, betaSection.row, ...(arena.vocab ? [tokenSection.row] : []), valueSection.row,
  );
  panel = {
    inputs, layer1: layer1.cells, layer2: layer2.cells, acceptFill, beta, tokens,
    acceptValue: acceptSection.value, betaValue: betaSection.value, valueValue: valueSection.value,
    layer1Value: layer1Section.value, layer2Value: layer2Section.value, inputValue: inputSection.value,
    tokenValue: tokenSection.value, inputRows: [],
  };
}

function paintCells(cells, values) {
  cells.forEach((cell, index) => {
    const value = clamp(num(values && values[index]) ?? 0, -1, 1);
    const hue = value >= 0 ? "var(--positive)" : "var(--negative)";
    cell.style.background = `color-mix(in srgb, ${hue} ${Math.round(Math.abs(value) * 100)}%, transparent)`;
  });
}

function renderNetwork() {
  if (!panel) buildNetworkPanel();
  const activations = net.activations || {};
  const inputs = activations.input || [];
  if (panel.inputRows.length !== inputs.length) {
    panel.inputs.replaceChildren();
    panel.inputRows = inputs.map((_, index) => {
      const name = el("span", "name", index < INPUT_NAMES.length ? INPUT_NAMES[index] : `message feature ${index - INPUT_NAMES.length + 1}`);
      const track = el("span", "track");
      const fill = el("span", "fill");
      track.append(fill);
      const number = el("span", "num");
      panel.inputs.append(name, track, number);
      return { fill, number };
    });
  }
  inputs.forEach((value, index) => {
    const row = panel.inputRows[index];
    row.fill.style.width = `${clamp(Math.abs(value) * 50, 0, 100)}%`;
    row.number.textContent = format(value, 2);
  });
  paintCells(panel.layer1, activations.layer1);
  paintCells(panel.layer2, activations.layer2);

  $("net-agent").textContent = net.agent === null ? "" : `agent ${net.agent} · ${net.action ?? ""} in round ${net.round ?? 0}`;
  panel.acceptFill.style.width = `${clamp((net.accept ?? 0) * 100, 0, 100)}%`;
  panel.acceptValue.textContent = percent(net.accept);
  panel.valueValue.textContent = format(net.value, 2);
  panel.layer1Value.textContent = activations.layer1 ? `mean |a| ${format(meanAbs(activations.layer1), 2)}` : "";
  panel.layer2Value.textContent = activations.layer2 ? `mean |a| ${format(meanAbs(activations.layer2), 2)}` : "";
  drawBeta();
  drawTokens();
}

function meanAbs(values) {
  return values.reduce((sum, value) => sum + Math.abs(value), 0) / Math.max(values.length, 1);
}

function drawBeta() {
  if (!net.beta) return;
  const [alpha, beta] = net.beta;
  panel.betaValue.textContent = `mean ${percent(alpha / (alpha + beta), 0)} to itself`;
  const points = [];
  let peak = 0;
  for (let i = 0; i <= 60; i += 1) {
    const x = i / 60;
    const density = Math.pow(Math.max(x, 1e-4), alpha - 1) * Math.pow(Math.max(1 - x, 1e-4), beta - 1);
    points.push(density);
    peak = Math.max(peak, density);
  }
  const path = points.map((density, index) => {
    const x = 10 + (index / 60) * 220;
    const y = 58 - (density / (peak || 1)) * 44;
    return `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join("");
  const area = `${path}L230 58L10 58Z`;
  const last = arena.duel && arena.duel.moves.length ? arena.duel.moves[arena.duel.moves.length - 1] : null;
  const pick = last && last.action === "offer" ? 10 + (1 - last.to_partner) * 220 : null;
  panel.beta.innerHTML =
    `<path class="area" d="${area}"/><path class="curve" d="${path}"/>` +
    `<line class="axis" x1="10" y1="58" x2="230" y2="58"/>` +
    (pick === null ? "" : `<line class="pick" x1="${pick.toFixed(1)}" y1="8" x2="${pick.toFixed(1)}" y2="58"/>`) +
    `<text x="10" y="68">keeps 0%</text><text x="196" y="68">100%</text>`;
}

function drawTokens() {
  if (!arena.vocab || !panel.tokens) return;
  const probabilities = net.messages || [];
  panel.tokenValue.textContent = net.chosen === null || net.chosen === undefined ? "" : `said m${net.chosen}`;
  panel.tokens.replaceChildren(...probabilities.map((probability, index) => {
    const token = el("div", `token${index === net.chosen ? " chosen" : ""}`);
    const bar = el("div", "bar");
    bar.style.height = `${clamp(probability * 100, 1, 100)}%`;
    token.append(bar, el("div", "name", `m${index}`));
    return token;
  }));
}

// =====================================================================================================
// History: charts, tiles, balances, feed (events.jsonl)
// =====================================================================================================

const history = {
  events: [], offset: 0, cursor: -1, mode: "live", hover: null, maxTokens: 0, connection: "connecting",
};
const feed = { shown: 0, cards: [] };
const CHARTS = [
  { id: "chart-population", key: "population", minTop: 4 },
  { id: "chart-gini", key: "gini", minTop: 0.1 },
];
let historyTimer = null;
let frame = 0;

async function pollEvents() {
  let delay = EVENTS_POLL_MS;
  try {
    const response = await fetch(`/api/events?offset=${history.offset}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    $("run-name").textContent = data.run;
    if (data.reset) resetHistory();
    history.offset = data.offset;
    ingest(data.events);
    history.connection = data.waiting ? "waiting" : "ok";
    if (data.more) delay = 0;
  } catch (error) {
    history.connection = "error";
    delay = EVENTS_POLL_MS * 3;
  }
  renderStatus();
  setTimeout(pollEvents, delay);
}

function ingest(events) {
  let added = false;
  for (const event of events) {
    if (!event || typeof event !== "object" || !Number.isInteger(event.generation)) continue;
    history.events.push(event);
    for (const agent of balancesOf(event)) history.maxTokens = Math.max(history.maxTokens, agent.tokens);
    appendTableRow(event);
    added = true;
  }
  if (!added) return;
  if (history.mode === "live") history.cursor = history.events.length - 1;
  render();
}

function resetHistory() {
  Object.assign(history, { events: [], cursor: -1, mode: "live", hover: null, maxTokens: 0 });
  clearInterval(historyTimer);
  feed.shown = 0;
  feed.cards = [];
  $("feed").replaceChildren($("feed-empty"));
  $("table-body").replaceChildren();
  render();
}

const historyEvent = () => history.events[history.cursor];
const balancesOf = (event) => (Array.isArray(event.token_balances) ? event.token_balances.filter((a) => a && num(a.tokens) !== null) : []);

function indexOfGeneration(generation) {
  let low = 0, high = history.events.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (history.events[mid].generation <= generation) low = mid; else high = mid - 1;
  }
  return low;
}

function setHistoryMode(mode) {
  history.mode = mode;
  clearInterval(historyTimer);
  historyTimer = null;
  if (mode === "live") history.cursor = history.events.length - 1;
  if (mode === "playing") historyTimer = setInterval(advanceHistory, 200);
  render();
}

function advanceHistory() {
  if (history.cursor >= history.events.length - 1) setHistoryMode("live");
  else { history.cursor += 1; render(); }
}

function jumpTo(generation) {
  if (!history.events.length || !Number.isFinite(generation)) return;
  history.cursor = indexOfGeneration(generation);
  setHistoryMode("paused");
}

function render() {
  if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawHistory(); });
}

function drawHistory() {
  const event = historyEvent();
  renderControls();
  renderStatus();
  renderTiles(event);
  drawCharts();
  renderBalances(event);
  syncFeed();
}

function renderControls() {
  const count = history.events.length;
  const first = count ? history.events[0].generation : 1;
  const last = count ? history.events[count - 1].generation : 1;
  const scrubber = $("scrubber");
  scrubber.min = first;
  scrubber.max = last;
  scrubber.disabled = !count;
  if (count) scrubber.value = history.events[history.cursor].generation;
  $("jump-to").min = first;
  $("jump-to").max = last;
  $("jump-to").placeholder = count ? `${first}–${last}` : "";
  $("hist-play").textContent = history.mode === "paused" ? "Play" : "Pause";
  $("hist-play").disabled = !count;
  $("hist-live").setAttribute("aria-pressed", String(history.mode === "live"));
  $("play").textContent = live.playing ? "Pause" : "Play";
  $("lag").textContent = live.queue.length > 12 ? `${live.queue.length} moves behind` : "";
  $("catchup").disabled = live.queue.length <= 12;
}

function renderStatus() {
  let name, icon, text;
  if (live.connection === "error" || history.connection === "error") [name, icon, text] = ["error", "⚠", "Can't reach the dashboard server, retrying"];
  else if (!live.playing) [name, icon, text] = ["paused", "❚❚", "Arena paused"];
  else if (live.waiting && !arena.agents.size) [name, icon, text] = ["waiting", "●", "Waiting for a live run"];
  else [name, icon, text] = ["live", "●", live.queue.length > CATCHUP_AT ? "Live, catching up" : "Live"];
  const status = $("status");
  status.dataset.state = name;
  status.querySelector(".icon").textContent = icon;
  $("status-text").textContent = text;
  const newest = history.events[history.events.length - 1];
  const logged = newest ? Date.parse(newest.time) : NaN;
  $("freshness").textContent = newest ? `newest generation ${newest.generation}${Number.isFinite(logged) ? `, logged ${ago(logged)}` : ""}` : "";
  $("arena-sub").textContent = arena.agents.size ? `${arena.agents.size} agents alive` : "";
}

function renderTiles(event) {
  const newest = history.events[history.events.length - 1];
  $("tile-generation").textContent = event ? String(event.generation) : String(arena.generation || "—");
  $("tile-generation-sub").textContent = event && newest ? (event === newest ? "newest logged" : `newest logged: ${newest.generation}`) : "";
  $("tile-population").textContent = arena.agents.size ? String(arena.agents.size) : event ? format(event.population, 0) : "—";
  $("tile-population-sub").textContent = event ? `${format(event.births, 0)} born, ${format(event.deaths, 0)} died last generation` : "";
  $("tile-gini").textContent = event ? format(event.gini, 3) : "—";
  $("tile-generosity").textContent = event ? percent(event.mean_generosity) : "—";
  $("tile-deals").textContent = arena.deals ? String(arena.deals) : "—";
  $("tile-deals-sub").textContent = arena.noDeals ? `${arena.noDeals} fell through` : "watched in the arena";
}

function niceStep(raw) {
  const exponent = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / exponent;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * exponent;
}

function drawCharts() {
  for (const chart of CHARTS) drawChart(chart);
}

function drawChart(chart) {
  const svg = $(chart.id);
  const width = Math.max(240, Math.round(svg.getBoundingClientRect().width));
  const height = 180;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("height", String(height));
  const events = history.events;
  if (!events.length) { svg.replaceChildren(); chart.geometry = null; return; }
  const m = { top: 10, right: 14, bottom: 26, left: 46 };
  const plotWidth = width - m.left - m.right;
  const plotHeight = height - m.top - m.bottom;
  const g0 = events[0].generation;
  const g1 = events[events.length - 1].generation;
  let peak = chart.minTop;
  for (const event of events) {
    const value = num(event[chart.key]);
    if (value !== null && value > peak) peak = value;
  }
  const yStep = niceStep(peak / 4);
  const top = Math.ceil(peak / yStep) * yStep;
  const decimals = yStep < 1 ? Math.ceil(-Math.log10(yStep)) : 0;
  const x = (g) => m.left + (g1 === g0 ? plotWidth / 2 : ((g - g0) / (g1 - g0)) * plotWidth);
  const y = (v) => m.top + plotHeight * (1 - v / top);
  chart.geometry = { g0, g1, left: m.left, plotWidth, width };

  const parts = [];
  for (let v = yStep; v <= top + yStep / 2; v += yStep) {
    parts.push(`<line class="grid" x1="${m.left}" x2="${width - m.right}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/>`);
  }
  for (let v = 0; v <= top + yStep / 2; v += yStep) {
    parts.push(`<text class="tick" x="${m.left - 8}" y="${y(v).toFixed(1)}" text-anchor="end" dominant-baseline="middle">${v.toFixed(decimals)}</text>`);
  }
  parts.push(`<line class="baseline" x1="${m.left}" x2="${width - m.right}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}"/>`);
  const xStep = Math.max(1, niceStep(Math.max(1, g1 - g0) / 6));
  for (let g = Math.ceil(g0 / xStep) * xStep; g <= g1; g += xStep) {
    parts.push(`<text class="tick" x="${x(g).toFixed(1)}" y="${height - 6}" text-anchor="middle">${g}</text>`);
  }
  let path = "";
  let penDown = false;
  for (const event of events) {
    const value = num(event[chart.key]);
    if (value === null) { penDown = false; continue; }
    path += `${penDown ? "L" : "M"}${x(event.generation).toFixed(1)} ${y(value).toFixed(1)}`;
    penDown = true;
  }
  parts.push(`<path class="line" d="${path}"/>`);
  if (history.hover !== null) {
    const hx = x(history.hover).toFixed(1);
    parts.push(`<line class="crosshair" x1="${hx}" x2="${hx}" y1="${m.top}" y2="${m.top + plotHeight}"/>`);
  }
  const shown = events[history.cursor];
  if (shown) {
    const cx = x(shown.generation).toFixed(1);
    parts.push(`<line class="cursor" x1="${cx}" x2="${cx}" y1="${m.top}" y2="${m.top + plotHeight}"/>`);
    const value = num(shown[chart.key]);
    if (value !== null) parts.push(`<circle class="dot" cx="${cx}" cy="${y(value).toFixed(1)}" r="4"/>`);
  }
  svg.innerHTML = parts.join("");
}

function generationAt(chart, pointer) {
  const geometry = chart.geometry;
  if (!geometry) return null;
  const rect = pointer.currentTarget.getBoundingClientRect();
  const px = (pointer.clientX - rect.left) * (geometry.width / rect.width);
  const t = geometry.g1 === geometry.g0 ? 0 : clamp((px - geometry.left) / geometry.plotWidth, 0, 1);
  return history.events[indexOfGeneration(Math.round(geometry.g0 + t * (geometry.g1 - geometry.g0)))].generation;
}

function showTooltip(pointer, event) {
  const tip = $("tooltip");
  const row = (value, label) => {
    const line = el("div", "tip-row");
    line.append(el("strong", "", value), el("span", "secondary", label));
    return line;
  };
  tip.replaceChildren(
    el("div", "tip-title", `Generation ${event.generation}`),
    row(format(event.population, 0), "living agents"),
    row(format(event.gini, 3), "Gini"),
    row(percent(event.mean_generosity), "mean generosity"),
  );
  tip.hidden = false;
  tip.style.left = `${Math.min(pointer.clientX + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
  tip.style.top = `${Math.min(pointer.clientY + 14, window.innerHeight - tip.offsetHeight - 8)}px`;
}

function renderBalances(event) {
  const container = $("balances");
  if (!event) { container.replaceChildren(el("p", "empty", "Waiting for the first generation.")); return; }
  const agents = balancesOf(event).sort((a, b) => b.tokens - a.tokens);
  const total = agents.reduce((sum, agent) => sum + agent.tokens, 0);
  $("balances-caption").textContent =
    `Generation ${event.generation}: ${agents.length} living agents holding ${format(total, 1)} tokens, richest first.`;
  container.replaceChildren(...agents.map((agent) => {
    const row = el("div", "bar-row");
    const track = el("div", "bar-track");
    const bar = el("div", "bar");
    bar.style.setProperty("--fraction", String(history.maxTokens > 0 ? Math.max(0, agent.tokens) / history.maxTokens : 0));
    track.append(bar, el("span", "bar-value", format(agent.tokens, 1)));
    row.append(el("span", "bar-label", `agent ${agent.agent_id}`), track);
    return row;
  }));
}

function syncFeed() {
  const list = $("feed");
  const target = history.cursor + 1;
  const atTop = list.scrollTop < 40;
  while (feed.shown < target) { list.prepend(cardFor(feed.shown)); feed.shown += 1; }
  while (feed.shown > target) { feed.shown -= 1; cardFor(feed.shown).remove(); }
  list.querySelector(".feed-item.current")?.classList.remove("current");
  if (target > 0) cardFor(target - 1).classList.add("current");
  $("feed-empty").hidden = target > 0;
  if (atTop) list.scrollTop = 0;
}

const cardFor = (index) => (feed.cards[index] ??= buildCard(history.events[index]));

function buildCard(event) {
  const card = el("article", "feed-item");
  const head = el("div", "item-head");
  head.append(el("strong", "", `Generation ${event.generation}`));
  card.append(head);
  const negotiation = event.negotiation;
  if (!negotiation || typeof negotiation !== "object") {
    head.append(el("span", "muted", "no negotiation this generation"));
    return card;
  }
  const ids = Array.isArray(negotiation.agent_ids) ? negotiation.agent_ids : ["?", "?"];
  const [icon, label] = OUTCOMES[negotiation.outcome] ?? ["•", String(negotiation.outcome)];
  const badge = el("span", "outcome");
  badge.dataset.outcome = String(negotiation.outcome);
  badge.append(el("span", "icon", icon), el("span", "", label));
  head.append(el("span", "muted", `economy round ${negotiation.economy_round}`), badge);

  let summary = `agent ${ids[0]} vs agent ${ids[1]}`;
  if (negotiation.outcome === "agreement" && Array.isArray(negotiation.split)) {
    summary += ` · agreed in round ${negotiation.negotiation_rounds} · ${negotiation.split[0]}% / ${negotiation.split[1]}%` +
      ` of a ${format(negotiation.surplus, 3)}-token surplus · price ${format(negotiation.price, 3)}`;
  } else {
    summary += ` · ${label} after ${negotiation.negotiation_rounds} rounds`;
  }
  card.append(el("p", "item-summary", summary));
  const steps = el("ol", "steps");
  for (const step of Array.isArray(negotiation.steps) ? negotiation.steps : []) steps.append(el("li", "", describeStep(step, ids)));
  card.append(steps);
  if (Array.isArray(negotiation.transcript)) {
    const raw = el("details", "raw");
    raw.append(el("summary", "", "Env transcript"), el("pre", "", negotiation.transcript.join("\n")));
    card.append(raw);
  }
  return card;
}

function describeStep(step, ids) {
  const partner = step.agent_id === ids[0] ? ids[1] : ids[0];
  const prefix = `round ${step.negotiation_round}: agent ${step.agent_id}`;
  switch (step.action) {
    case "talk": return `${prefix} says m${step.message}`;
    case "offer": return `${prefix} offers ${percent(step.offer_to_partner)} of the surplus to agent ${partner}`;
    case "accept": return `${prefix} accepts`;
    case "reject": return `${prefix} rejects the final offer`;
    default: return `${prefix} ${step.action}`;
  }
}

function appendTableRow(event) {
  const row = el("tr");
  for (const [value, digits] of [[event.generation, 0], [event.population, 0], [event.births, 0], [event.deaths, 0], [event.gini, 3]]) {
    row.append(el("td", "", format(value, digits)));
  }
  row.append(el("td", "", percent(event.mean_generosity)));
  $("table-body").append(row);
}

// =====================================================================================================
// Wiring
// =====================================================================================================

$("play").addEventListener("click", () => {
  live.playing = !live.playing;
  live.nextAt = performance.now();
  renderControls();
  renderStatus();
});
$("speed").addEventListener("change", (event) => { live.speed = Number(event.target.value); });
$("catchup").addEventListener("click", () => {
  const now = performance.now();
  while (live.queue.length > 8) applyLive(live.queue.shift(), now);
  live.nextAt = now;
  renderControls();
});
$("hist-play").addEventListener("click", () => {
  if (!history.events.length) return;
  if (history.mode !== "paused") setHistoryMode("paused");
  else setHistoryMode(history.cursor >= history.events.length - 1 ? "live" : "playing");
});
$("hist-live").addEventListener("click", () => { if (history.events.length) setHistoryMode("live"); });
$("scrubber").addEventListener("input", (event) => jumpTo(Number(event.target.value)));
$("jump").addEventListener("submit", (event) => {
  event.preventDefault();
  jumpTo(Number($("jump-to").value));
});
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.target.closest("input, select, textarea, button, summary")) return;
  if (event.key === " ") { event.preventDefault(); $("play").click(); }
  else if (event.key === "ArrowLeft") { history.cursor = Math.max(0, history.cursor - 1); setHistoryMode("paused"); }
  else if (event.key === "ArrowRight") { history.cursor = Math.min(history.events.length - 1, history.cursor + 1); setHistoryMode("paused"); }
  else if (event.key === "l" || event.key === "L") { if (history.events.length) setHistoryMode("live"); }
});
for (const chart of CHARTS) {
  const svg = $(chart.id);
  svg.addEventListener("pointermove", (pointer) => {
    const generation = generationAt(chart, pointer);
    if (generation === null) return;
    if (generation !== history.hover) { history.hover = generation; drawCharts(); }
    showTooltip(pointer, history.events[indexOfGeneration(generation)]);
  });
  svg.addEventListener("pointerleave", () => { history.hover = null; $("tooltip").hidden = true; drawCharts(); });
  svg.addEventListener("click", (pointer) => {
    const generation = generationAt(chart, pointer);
    if (generation !== null) jumpTo(generation);
  });
}
window.addEventListener("resize", render);
setInterval(renderStatus, 1000);

function loop(now) {
  requestAnimationFrame(loop);
  pump(now);
  drawArena(now);
}

buildNetworkPanel();
render();
requestAnimationFrame(loop);
pollLive();
pollEvents();
