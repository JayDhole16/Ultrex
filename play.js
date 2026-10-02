"use strict";
// The page holds no model and no history of its own: it renders whatever /api/state reports and sends the
// visitor's moves to /api/command. Every number shown came from a forward pass a moment earlier.

const POLL_MS = 600;
const INPUT_NAMES = ["balance", "need (valuation)", "offered to me", "offered to them", "round", "rounds left", "my turn"];
const SCENARIO_CONTROLS = ["seat_a", "seat_b", "need_a", "need_b", "max_rounds", "discount", "protocol", "observe_round", "topic"];

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const pct = (v, digits = 0) => (num(v) === null ? "—" : `${(v * 100).toFixed(digits)}%`);
const fixed = (v, digits) => (num(v) === null ? "—" : v.toFixed(digits));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;  // server text is data, never markup
  return node;
}

let state = null;
let panel = null;
let rosterKey = "";

async function command(body) {
  const response = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.ok) render(await response.json());
}

async function poll() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (response.ok) render(await response.json());
  } catch (error) {
    $("status-text").textContent = "Lost the server, retrying";
  }
  setTimeout(poll, POLL_MS);
}

function render(next) {
  state = next;
  const weights = state.weights || {};
  const source = state.checkpoint.path
    ? `${state.checkpoint.path} · sha256 ${state.checkpoint.sha256}`
    : `fresh random weights, seed ${weights.seed}`;
  const updated = weights.version
    ? ` · weights update ${weights.version}, ${fixed(weights.age, 1)}s ago`
    : "";
  $("provenance").textContent = `${source} · ${state.checkpoint.policy}${updated}`;
  renderScenario();
  renderStage();
  renderMoves();
  renderOutcome();
  renderNetwork();
  renderTraining();
  renderStatus();
  const language = state.language || {};
  $("language_status").textContent = language.available
    ? language.description || "Ready."
    : `Not available: ${language.reason || "not configured"}. The agents bargain in offers and tokens; English needs a language model wired in, and this machine has none.`;
}

function renderStatus() {
  const status = $("status");
  let name = "idle", text = "Ready";
  if (state.thinking) [name, text] = ["thinking", "Language model is thinking"];
  else if (state.turn) [name, text] = ["your-turn", "Your move"];
  else if (state.moves.length && !state.outcome) [name, text] = ["idle", "Agents deciding"];
  else if (state.training.running) [name, text] = ["training", "Learning live"];
  else if (state.outcome) [name, text] = ["done", "Negotiation finished"];
  if (state.training.running && name !== "training") text += " · agents learning live";
  status.dataset.state = name;
  $("status-text").textContent = text;
}

// ---- scenario ---------------------------------------------------------------------------------------

function renderScenario() {
  const roster = state.roster;
  const key = Object.keys(roster).join(",");
  if (key !== rosterKey) {
    rosterKey = key;
    for (const id of ["seat_a", "seat_b"]) {
      $(id).replaceChildren(...Object.entries(roster).map(([value, label]) => {
        const option = el("option", "", label);
        option.value = value;
        return option;
      }));
    }
    const opponents = [["self", "itself (self-play)"], ...Object.entries(roster).filter(([value]) => value !== "you" && value !== "llm")];
    $("train_opponent").replaceChildren(...opponents.map(([value, label]) => {
      const option = el("option", "", label);
      option.value = value;
      return option;
    }));
    $("train_opponent").value = "hardball";
  }
  const scenario = state.scenario;
  for (const id of SCENARIO_CONTROLS) {
    const control = $(id);
    if (document.activeElement === control) continue;  // never fight the hand on the slider
    if (control.type === "checkbox") control.checked = !scenario[id];  // the control reads "hide the clock"
    else control.value = scenario[id];
  }
  labelScenario();
}

function labelScenario() {
  $("need_a_v").textContent = `${Number($("need_a").value).toFixed(2)}x`;
  $("need_b_v").textContent = `${Number($("need_b").value).toFixed(2)}x`;
  $("max_rounds_v").textContent = `${$("max_rounds").value} rounds`;
  $("discount_v").textContent = `${Number($("discount").value).toFixed(2)}`;
  $("train_updates_v").textContent = Number($("train_updates").value) >= 85 ? "until stopped" : `${$("train_updates").value}`;
}

function pushScenario() {
  command({
    command: "scenario",
    scenario: {
      seat_a: $("seat_a").value,
      seat_b: $("seat_b").value,
      need_a: Number($("need_a").value),
      need_b: Number($("need_b").value),
      max_rounds: Number($("max_rounds").value),
      discount: Number($("discount").value),
      protocol: $("protocol").value,
      observe_round: $("observe_round").checked ? 0 : 1,
      topic: $("topic").value,
    },
  });
}

// ---- the stage --------------------------------------------------------------------------------------

function renderStage() {
  const roster = state.roster;
  const seats = [state.scenario.seat_a, state.scenario.seat_b];
  const needs = [state.scenario.need_a, state.scenario.need_b];
  seats.forEach((seat, index) => {
    const box = $(`seat_${"ab"[index]}_box`);
    $(`seat_${"ab"[index]}_name`).textContent = roster[seat] ?? seat;
    $(`seat_${"ab"[index]}_stat`).textContent = `values tokens ${needs[index].toFixed(2)}x`;
    box.classList.toggle("you", seat === "you");
    box.classList.toggle("acting", Boolean(state.turn) && state.turn.seat === index && seat !== "you");
  });

  const table = state.table;
  $("table").hidden = !table;
  if (table) {
    const total = Math.max(1, table.keeps + table.gives);
    $("table_who").textContent = `on the table, from ${roster[table.from] ?? table.from}`;
    $("table_split").textContent = `${table.keeps} / ${table.gives}`;
    $("table_keep").style.flex = String(table.keeps / total);
    $("table_give").style.flex = String(table.gives / total);
  }

  const turn = state.turn;
  $("your_move").hidden = !turn;
  if (!turn) return;
  const offered = Math.round(turn.offered_to_you);
  $("move_context").textContent = turn.can_talk
    ? `Round ${turn.round}. Both sides speak before anyone moves.`
    : `Round ${turn.round}, ${turn.rounds_remaining} left. ${turn.can_accept ? `They offered you ${offered} of ${state.pool}.` : "Nothing on the table yet: open with an offer."}`;
  $("talk_controls").hidden = !turn.can_talk;
  $("offer_controls").hidden = Boolean(turn.can_talk);
  if (turn.can_talk) {
    const vocab = state.checkpoint.message_vocab || 0;
    $("talk_buttons").replaceChildren(...Array.from({ length: vocab }, (_, token) => {
      const button = el("button", "", `say m${token}`);
      button.addEventListener("click", () => command({ command: "move", turn: state?.turn?.id, move: { action: "message", message: token } }));
      return button;
    }));
    return;
  }
  $("send_offer").hidden = !turn.can_offer;
  $("accept").hidden = !turn.can_accept;
  $("reject").hidden = !(turn.can_accept && !turn.can_offer);
  labelShare();
}

function labelShare() {
  const keep = Number($("share").value);
  const pool = state ? state.pool : 100;
  $("share_v").textContent = `${keep} / ${pool - keep}`;
}

// What the language model weighed before it moved: each option's mean log-probability after its own
// sentence, shown relative to the favourite. The tallest bar is the move it made.
function weighedStrip(options, seconds) {
  const top = Math.max(...options.map((option) => option.score));
  const weights = options.map((option) => Math.exp(option.score - top));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const strip = el("div", "weighed");
  strip.append(el("span", "label", "weighed"));
  options.forEach((option, index) => {
    const name = option.option ?? String(Math.round(option.keep * 100));
    const column = el("div", `opt${option.score === top ? " chosen" : ""}`);
    const bar = el("div", "bar");
    bar.style.height = `${Math.max(2, 20 * weights[index])}px`;
    column.title = `${name}: ${Math.round((100 * weights[index]) / total)}% of its preference (mean log-prob ${option.score})`;
    column.append(bar, el("span", "name", name));
    strip.append(column);
  });
  if (num(seconds) !== null) strip.append(el("span", "label", ` ${seconds.toFixed(1)}s on this CPU`));
  return strip;
}

function renderMoves() {
  const roster = state.roster;
  $("moves").replaceChildren(...state.moves.map((move) => {
    const row = el("div", `move${move.who === "you" ? " you" : ""}`);
    const who = roster[move.who] ?? move.who;
    let what;
    if (move.kind === "talk") what = `${who} says m${move.message}`;
    else if (move.kind === "accept") what = `${who} accepts`;
    else if (move.kind === "reject") what = `${who} walks away`;
    else what = `${who} offers ${move.keeps} / ${move.gives}`;
    if (num(move.weights) !== null && state.weights?.live) what += ` · update ${move.weights}`;
    const body = el("span", "what");
    if (move.say) {  // a language model said something before moving
      body.append(el("em", "", `“${move.say}” `), el("span", "secondary", `→ ${what}`));
      if (move.considered?.length) body.append(weighedStrip(move.considered, move.seconds));
    } else {
      body.textContent = what;
    }
    row.append(el("span", "when", `round ${move.round}`), body);
    return row;
  }));
  if (state.thinking) {  // the server is mid-sentence: say so, with how long it has been at it
    const row = el("div", "move thinking");
    const seconds = state.thinking.elapsed ?? 0;
    const who = roster[state.thinking.who] ?? state.thinking.who;
    const doing = state.thinking.loading
      ? "is loading into memory for its first turn (once per session, about ten seconds)"
      : "is writing its argument and weighing its price";
    const body = el("span", "what", `${who} ${doing}, ${seconds.toFixed(0)}s`);
    body.append(el("span", "ellipsis"));
    row.append(el("span", "when", ""), body);
    $("moves").append(row);
  }
  $("moves").scrollTop = $("moves").scrollHeight;
}

function renderOutcome() {
  const outcome = state.outcome;
  $("outcome").hidden = !outcome;
  if (!outcome) return;
  const deal = outcome.outcome === "agreement";
  $("outcome").className = `outcome ${deal ? "deal" : "nodeal"}`;
  $("outcome_headline").textContent = deal
    ? `Deal in round ${outcome.rounds}: ${outcome.tokens[0]} / ${outcome.tokens[1]}`
    : `No deal (${outcome.outcome})`;
  $("outcome_detail").textContent = deal
    ? `payoffs ${outcome.payoffs[0].toFixed(1)} and ${outcome.payoffs[1].toFixed(1)}, after each side's private valuation and the cost of delay`
    : "both sides walk away with nothing";
  $("outcome_transcript").textContent = outcome.transcript.join("\n");
}

// ---- the network ------------------------------------------------------------------------------------

function buildPanel() {
  const cells = (count) => {
    const grid = el("div", "cells");
    const list = Array.from({ length: count }, () => el("div", "cell"));
    grid.append(...list);
    return { grid, list };
  };
  const section = (title, body) => {
    const row = el("div", "net-row");
    const head = el("div", "head");
    const value = el("span", "value");
    head.append(el("span", "", title), value);
    row.append(head, body);
    return { row, value };
  };
  const inputs = el("div", "inputs");
  const layer1 = cells(64);
  const layer2 = cells(64);
  const accept = el("div", "prob-track");
  const acceptFill = el("div", "prob-fill");
  accept.append(acceptFill);
  const beta = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  beta.setAttribute("class", "beta");
  beta.setAttribute("viewBox", "0 0 240 70");
  beta.setAttribute("height", "70");
  const tokens = el("div", "tokens");
  const scale = el("div", "scale");
  scale.append(el("span", "", "−1"), el("div", "ramp"), el("span", "", "+1"));

  const sections = {
    inputs: section("What it sees", inputs),
    layer1: section("Hidden layer 1 · 64 units", layer1.grid),
    layer2: section("Hidden layer 2 · 64 units", layer2.grid),
    accept: section("Accept probability", accept),
    beta: section("Offer it would make", beta),
    tokens: section("Message probabilities", tokens),
    value: section("Value estimate", el("div", "caption", "what it expects this negotiation to be worth")),
  };
  $("net").replaceChildren(
    sections.inputs.row, sections.layer1.row, sections.layer2.row, scale,
    sections.accept.row, sections.beta.row,
    ...(state.checkpoint.message_vocab ? [sections.tokens.row] : []), sections.value.row,
  );
  panel = { inputs, rows: [], layer1: layer1.list, layer2: layer2.list, acceptFill, beta, tokens, sections };
}

function paint(cells, values) {
  cells.forEach((cell, index) => {
    const value = clamp(num(values && values[index]) ?? 0, -1, 1);
    const hue = value >= 0 ? "var(--positive)" : "var(--negative)";
    cell.style.background = `color-mix(in srgb, ${hue} ${Math.round(Math.abs(value) * 100)}%, transparent)`;
  });
}

function renderNetwork() {
  if (!panel) buildPanel();
  const net = state.network;
  const trained = state.scenario ? [state.scenario.seat_a, state.scenario.seat_b].filter((seat) => seat.startsWith("agent_")) : [];
  $("net_caption").textContent = net
    ? `${state.roster[net.who] ?? net.who}, deciding in round ${net.round} with the weights from update ${net.weights ?? 0}. Every number is a forward pass on the observation in front of it.`
    : trained.length
      ? "Waiting for the trained network's first move in this negotiation."
      : "No trained network is seated in this negotiation, so there are no layers to show. Put a trained agent in a seat to watch it think.";
  $("net").hidden = !net;  // an empty panel would show last negotiation's numbers, or none that mean anything
  if (!net) return;

  const features = net.features || [];
  if (panel.rows.length !== features.length) {
    panel.inputs.replaceChildren();
    panel.rows = features.map((_, index) => {
      const name = el("span", "name", index < INPUT_NAMES.length ? INPUT_NAMES[index] : `message feature ${index - INPUT_NAMES.length + 1}`);
      const track = el("span", "track");
      const fill = el("span", "fill");
      track.append(fill);
      const number = el("span", "num");
      panel.inputs.append(name, track, number);
      return { fill, number };
    });
  }
  features.forEach((value, index) => {
    panel.rows[index].fill.style.width = `${clamp(Math.abs(value) * 50, 0, 100)}%`;
    panel.rows[index].number.textContent = fixed(value, 2);
  });
  paint(panel.layer1, net.layer1);
  paint(panel.layer2, net.layer2);
  panel.acceptFill.style.width = `${clamp(net.accept_prob * 100, 0, 100)}%`;
  panel.sections.accept.value.textContent = pct(net.accept_prob, 1);
  panel.sections.value.value.textContent = fixed(net.value, 2);
  drawBeta(net.beta);
  drawTokens(net.message_probs);
}

function drawBeta([alpha, beta]) {
  panel.sections.beta.value.textContent = `mean ${pct(alpha / (alpha + beta))} to itself`;
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
  panel.beta.innerHTML =
    `<path class="area" d="${path}L230 58L10 58Z"/><path class="line" d="${path}"/>` +
    `<line class="axis" x1="10" y1="58" x2="230" y2="58"/>` +
    `<text x="10" y="68">keeps 0%</text><text x="196" y="68">100%</text>`;
}

function drawTokens(probabilities) {
  if (!state.checkpoint.message_vocab || !probabilities) return;
  panel.tokens.replaceChildren(...probabilities.map((probability, index) => {
    const token = el("div", "token");
    const bar = el("div", "bar");
    bar.style.height = `${clamp(probability * 100, 1, 100)}%`;
    token.append(bar, el("div", "name", `m${index}`));
    return token;
  }));
}

// ---- training on stage ------------------------------------------------------------------------------

function renderTraining() {
  const training = state.training;
  const history = training.history || [];
  const last = history[history.length - 1];
  const selfPlay = training.opponent === "self";
  $("train_start").disabled = training.running;
  $("train_stop").disabled = !training.running;
  $("reload").hidden = !(state.weights && state.weights.can_reset);
  const against = selfPlay ? "self-play" : `against ${state.roster[training.opponent] ?? training.opponent}`;
  const reward = selfPlay ? `${fixed(last?.reward, 1)} / ${fixed(last?.reward_b, 1)}` : fixed(last?.reward, 1);
  let status;
  if (history.length && last.update > 0) {
    status = `${training.running ? "Learning" : "Learned"} ${against}: update ${last.update} · ${last.steps.toLocaleString()} env steps`
      + ` · reward ${reward} · deals in ${fixed(last.rounds, 1)} rounds · opens at ${pct(last.opening_offer)}`
      + ` · ${num(last.walk_away) === null ? "no walk-away price yet" : `walks away below ${pct(last.walk_away)}`}`
      + ` · takes a 20-token lowball ${pct(last.lowball_accept)} of the time`;
  } else if (training.running) {
    status = `Starting ${against}: collecting the first 4,096 moves.`;
  } else {
    status = training.error ? `Stopped on an error: ${training.error}` : "Not learning. Press Train, or Start from scratch.";
  }
  $("train_status").textContent = status;
  drawReward(history, selfPlay);
  drawLowball(history);
  drawBehaviour(history);
}

function plot(svg, height, series, ticks, labels) {
  if (!series[0].points.length) { svg.replaceChildren(); return; }
  const width = 260, m = { top: 8, right: 8, bottom: 16, left: 32 };
  const count = series[0].points.length;
  const x = (i) => m.left + (count === 1 ? 0 : (i / (count - 1)) * (width - m.left - m.right));
  const y = (v) => m.top + (1 - clamp((v - ticks.low) / (ticks.high - ticks.low || 1), 0, 1)) * (height - m.top - m.bottom);
  const parts = [];
  for (const level of ticks.marks) {
    parts.push(`<line class="grid" x1="${m.left}" x2="${width - m.right}" y1="${y(level).toFixed(1)}" y2="${y(level).toFixed(1)}"/>`);
    parts.push(`<text x="2" y="${(y(level) + 3).toFixed(1)}">${ticks.format(level)}</text>`);
  }
  for (const line of series) {
    let path = "", pen = false;  // a missing value lifts the pen: a gap, never a made-up point
    line.points.forEach((value, index) => {
      if (num(value) === null) { pen = false; return; }
      path += `${pen ? "L" : "M"}${x(index).toFixed(1)} ${y(value).toFixed(1)}`;
      pen = true;
    });
    if (path) parts.push(`<path class="line" style="stroke:${line.colour}" d="${path}"/>`);
  }
  labels.forEach((label, index) => parts.push(`<text x="${m.left + index * 96}" y="${height - 3}" style="fill:${label.colour}">${label.text}</text>`));
  svg.innerHTML = parts.join("");
}

function drawReward(history, selfPlay) {
  const measured = history.filter((point) => num(point.reward) !== null);  // update 0 has no games yet
  const a = measured.map((point) => point.reward);
  const b = measured.map((point) => num(point.reward_b) ?? 0);
  const all = selfPlay ? [...a, ...b] : a;
  const low = Math.min(...all, 0), high = Math.max(...all, 1);
  const series = [{ points: a, colour: "var(--series)" }];
  if (selfPlay) series.push({ points: b, colour: "var(--series-b)" });
  plot($("train_reward"), 90, series,
    { low, high, marks: [low, (low + high) / 2, high], format: (v) => v.toFixed(0) },
    selfPlay
      ? [{ text: "agent_0", colour: "var(--series)" }, { text: "agent_1", colour: "var(--series-b)" }]
      : [{ text: "payoff per negotiation", colour: "var(--muted)" }]);
}

function drawLowball(history) {
  plot($("train_lowball"), 80, [{ points: history.map((point) => num(point.lowball_accept) ?? 0), colour: "var(--series)" }],
    { low: 0, high: 1, marks: [0, 0.5, 1], format: (v) => `${Math.round(v * 100)}%` },
    [{ text: "accepts 20 of 100", colour: "var(--muted)" }]);
}

function drawBehaviour(history) {
  plot($("train_curve"), 100, [
    { points: history.map((point) => point.opening_offer), colour: "var(--series)" },
    { points: history.map((point) => point.walk_away), colour: "var(--series-b)" },
  ], { low: 0, high: 1, marks: [0, 0.5, 1], format: (v) => `${Math.round(v * 100)}%` },
    [{ text: "opens at", colour: "var(--series)" }, { text: "walks away below", colour: "var(--series-b)" }]);
}

// ---- wiring -----------------------------------------------------------------------------------------

for (const id of SCENARIO_CONTROLS) {
  $(id).addEventListener("input", labelScenario);
  $(id).addEventListener("change", pushScenario);
}
$("train_updates").addEventListener("input", labelScenario);
$("share").addEventListener("input", labelShare);
$("start").addEventListener("click", () => command({ command: "start" }));
$("send_offer").addEventListener("click", () => command({ command: "move", turn: state?.turn?.id, move: { action: "offer", share: Number($("share").value) / 100 } }));
$("accept").addEventListener("click", () => command({ command: "move", turn: state?.turn?.id, move: { action: "accept" } }));
$("reject").addEventListener("click", () => command({ command: "move", turn: state?.turn?.id, move: { action: "reject" } }));
$("train_start").addEventListener("click", () => command({
  command: "train_start", opponent: $("train_opponent").value,
  updates: Number($("train_updates").value) >= 85 ? 0 : Number($("train_updates").value),
}));
$("train_stop").addEventListener("click", () => command({ command: "train_stop" }));
$("reload").addEventListener("click", () => command({ command: "reload" }));
$("fresh").addEventListener("click", () => command({ command: "fresh" }));

poll();
