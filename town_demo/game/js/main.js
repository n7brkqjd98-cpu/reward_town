// Reward Town overworld: boot, input, game loop, NPC movement, tick pacing, talk/log/mind/pause flows.

import * as api from "./api.js";
import * as ui from "./ui.js";
import { PLAYER_ID, TOPIC_LABELS, runTalk, watchConversation } from "./talk.js";
import { runHelpRequest, runTechnique, showSkills, showWorkLog, siteActions, toolActions } from "./tools.js";
import { drawArch } from "./arch.js";
import { drawLayers, findPath, isBlocked, loadMap } from "./map.js";
import { drawCharacter, frontPortrait, loadImage, loadSprites } from "./sprites.js";

const TICK_MS = 6000;
const SPEEDS = [1, 2, 4];
const WALK_MS = 200;
const RUN_MS = 110;
const VIEW_W = 320;
const VIEW_H = 208;
const SCALE = 3;
const PLATE_RANGE = 3;
const PLAYER_SPRITES = ["adventurer", "heroine"];
const NAME_RE = /^[A-Za-z][A-Za-z .'-]{0,11}$/;
const KEYS = {
  ArrowUp: "up", KeyW: "up", ArrowDown: "down", KeyS: "down", ArrowLeft: "left", KeyA: "left",
  ArrowRight: "right", KeyD: "right", KeyZ: "a", Space: "a", Enter: "a", NumpadEnter: "a", KeyX: "b",
  Escape: "b", KeyL: "log", KeyJ: "work", KeyK: "skills", KeyT: "speed", KeyP: "pause",
};
const DIRS = new Set(["up", "down", "left", "right"]);
const DELTA = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const OPPOSITE = { up: "down", down: "up", left: "right", right: "left" };

const $ = (sel) => document.querySelector(sel);
const ctx = $("#world").getContext("2d");
ctx.imageSmoothingEnabled = false;

const G = {
  map: null,
  sheets: {},
  bubbles: {},
  phase: "title", // "title" | "intro" | "playing" | "ended"
  meta: null,
  locations: [],
  locById: {},
  people: {}, // id -> {name, age, sprite}; includes the player
  npcs: new Map(), // id -> {id, x, y, dir, step, path, rec}
  player: { x: 0, y: 0, dir: "up", step: null, inside: null },
  playerState: null,
  frame: null,
  conversations: [],
  tools: [],
  toolById: {},
  toolUses: new Map(), // id -> tool-use record
  project: null, // the heart arch: {id, name, location, tiles, hours, stages}
  revealed: new Set(), // "c:<id>" / "u:<id>": records already revealed to the player in a dialog
  speedIdx: 0,
  tickElapsed: 0,
  stepInFlight: false,
  busy: false, // a talk/encounter/log/pause flow owns the game (pauses the clock)
  held: [],
  shift: false,
  panelCursor: 0,
  alert: null, // npc id showing the "!" bubble
  labelSig: "",
};

const speed = () => SPEEDS[G.speedIdx];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const firstName = (id) => G.people[id]?.name.split(" ")[0] ?? id;
const hhmm = (t) => `${String(Math.floor((t * 10) / 60)).padStart(2, "0")}:${String((t * 10) % 60).padStart(2, "0")}`;

function stagesDone() {
  const ids = new Set(G.project.stages.map((s) => s.id));
  let done = 0;
  for (const r of G.toolUses.values()) if (ids.has(r.stage) && r.stage_done_t !== null) done += 1;
  return done;
}

// ---------- positions ----------
function lerpPos(c) {
  if (!c.step) return { x: c.x, y: c.y };
  const k = clamp((performance.now() - c.step.t0) / c.step.dur, 0, 1);
  return { x: c.step.fx + (c.step.tx - c.step.fx) * k, y: c.step.fy + (c.step.ty - c.step.fy) * k };
}

const npcWalking = (n) => Boolean(n.step) || n.path.length > 0;
const npcVisible = (n) => npcWalking(n) || !n.rec.inside;

function areaLoc(x, y) {
  for (const loc of G.locations) {
    if (!loc.area) continue;
    const [ax, ay, aw, ah] = loc.area;
    if (x >= ax && x < ax + aw && y >= ay && y < ay + ah) return loc.id;
  }
  return null;
}

function playerLoc() {
  const p = G.player;
  return p.inside ?? areaLoc(p.x, p.y);
}

function doorAt(x, y) {
  return G.locations.find((l) => l.door && l.door[0] === x && l.door[1] === y) ?? null;
}

function npcStandingAt(x, y) {
  for (const n of G.npcs.values()) if (npcVisible(n) && !npcWalking(n) && n.x === x && n.y === y) return n;
  return null;
}

// ---------- frames ----------
function initNpcs(frame) {
  G.npcs.clear();
  for (const [id, rec] of Object.entries(frame.agents)) {
    G.npcs.set(id, { id, x: rec.x, y: rec.y, dir: "down", step: null, path: [], rec });
  }
  G.frame = frame;
}

function applyFrame(frame) {
  G.frame = frame;
  for (const [id, rec] of Object.entries(frame.agents)) {
    const n = G.npcs.get(id);
    n.rec = rec;
    const from = n.step ? { x: n.step.tx, y: n.step.ty } : { x: n.x, y: n.y };
    const last = n.path.length ? n.path[n.path.length - 1] : from;
    if (last.x === rec.x && last.y === rec.y) continue;
    const path = findPath(from, { x: rec.x, y: rec.y });
    if (path === null) {
      n.x = rec.x;
      n.y = rec.y;
      n.step = null;
      n.path = [];
    } else {
      n.path = path;
    }
  }
  renderPanel();
}

function absorb(resp) {
  if (resp.player) G.playerState = resp.player;
  for (const c of resp.new_conversations ?? []) {
    G.conversations.push(c);
    if (!c.participants.includes(PLAYER_ID)) {
      const label = TOPIC_LABELS[c.topic] ?? c.topic;
      ui.toast(`💬 ${firstName(c.participants[0])} talked with ${firstName(c.participants[1])} ` +
        `(${label}: ${c.effect}) — L to watch`);
    }
  }
  for (const r of resp.tool_uses ?? []) {
    const prev = G.toolUses.get(r.id);
    G.toolUses.set(r.id, r);
    const tool = G.toolById[r.tool];
    const mine = r.user === PLAYER_ID || r.helper === PLAYER_ID;
    const stage = r.stage !== null ? G.project.stages.find((s) => s.id === r.stage) : null;
    if (!prev && !mine) {
      ui.toast(stage
        ? `🏗 ${firstName(r.user)} started the ${stage.name} of the heart arch with the ${tool.name} — nobody asked, ` +
          "they just saw it"
        : `${tool.emoji} ${firstName(r.user)}${r.helper !== null ? ` + ${firstName(r.helper)}` : ""}: ` +
          `${tool.product} Q${r.quality} — J to view`);
    } else if (prev && prev.helper !== r.helper && r.helper !== PLAYER_ID) {
      ui.toast(`🤝 ${firstName(r.helper)} joined ${firstName(r.user)} at the ${tool.name}: ` +
        `Q${r.solo_quality}→Q${r.quality}`);
    }
    if (stage && r.stage_done_t !== null && (prev?.stage_done_t ?? null) === null) {
      const n = G.project.stages.length;
      ui.toast(`✔ ${firstName(r.user)}${r.helper !== null ? ` + ${firstName(r.helper)}` : ""} finished the ` +
        `${stage.name} (${stagesDone()}/${n})`);
      if (stagesDone() === n) ui.toast(`💝 The ${G.project.name} is finished!`);
    }
    for (const e of r.learned.slice(prev?.learned.length ?? 0)) {
      if (e.agent === PLAYER_ID) ui.toast(`📖 You learned a new way with the ${tool.name}!`);
      else if (e.by === PLAYER_ID && !G.revealed.has(`u:${r.id}`)) {
        ui.toast(`📖 ${firstName(e.agent)} learned your way with the ${tool.name}!`);
      }
    }
  }
  if (resp.frame) applyFrame(resp.frame);
  else renderPanel();
}

function flowCtx() {
  return {
    people: G.people, toolById: G.toolById, toolUses: G.toolUses, revealed: G.revealed, locById: G.locById,
    project: G.project,
    playerState: () => G.playerState, onResponse: absorb, process: processResponse,
  };
}

// ---------- server turns ----------
async function encounter(npcId) {
  G.alert = npcId;
  renderPanel();
  await sleep(600);
  const flash = $("#flash");
  flash.classList.remove("on");
  void flash.offsetWidth;
  flash.classList.add("on");
  await sleep(300);
  G.alert = null;
  renderPanel();
}

// Applies a step/choose/talk/tool response and runs every player turn it asks for (possibly several per tick).
async function processResponse(resp) {
  absorb(resp);
  for (;;) {
    if (resp.ended) {
      await showSummary(resp.summary);
      return;
    }
    const pending = resp.pending;
    if (!pending) return;
    const c = pending.context;
    if (pending.kind === "line") {
      if (c.initiator !== PLAYER_ID) await encounter(c.initiator);
      resp = await runTalk(pending, flowCtx());
    } else if (pending.kind === "technique") {
      resp = await runTechnique(pending, flowCtx());
    } else if (pending.kind === "help_request") {
      await encounter(c.user);
      resp = await runHelpRequest(pending, flowCtx());
    } else {
      throw new Error(`unknown pending turn ${pending.kind}`);
    }
  }
}

async function doStep() {
  const p = G.player;
  G.stepInFlight = true;
  try {
    const resp = await api.step({ x: p.x, y: p.y, loc: playerLoc() });
    G.tickElapsed = 0;
    G.stepInFlight = false;
    G.busy = true;
    await processResponse(resp);
  } catch (err) {
    G.tickElapsed = 0;
    ui.toast(`⚠ ${err.message}`);
  } finally {
    G.stepInFlight = false;
    G.busy = false;
  }
}

// ---------- movement ----------
function updatePlayer(now) {
  const p = G.player;
  if (p.step && now - p.step.t0 >= p.step.dur) {
    p.x = p.step.tx;
    p.y = p.step.ty;
    p.step = null;
    const door = doorAt(p.x, p.y);
    if (door && p.dir === "up") enterBuilding(door.id);
  }
  if (p.step || p.inside || ui.hasModal() || G.busy) return;
  const dir = G.held[G.held.length - 1];
  if (!dir) return;
  p.dir = dir;
  const tx = p.x + DELTA[dir][0];
  const ty = p.y + DELTA[dir][1];
  if (isBlocked(tx, ty) || npcStandingAt(tx, ty)) return;
  p.step = { fx: p.x, fy: p.y, tx, ty, t0: now, dur: G.shift ? RUN_MS : WALK_MS };
}

function updateNpcs(now) {
  const dur = 160 / speed();
  for (const n of G.npcs.values()) {
    if (n.step && now - n.step.t0 >= n.step.dur) {
      n.x = n.step.tx;
      n.y = n.step.ty;
      n.step = null;
      if (!n.path.length) n.dir = "down";
    }
    if (!n.step && n.path.length) {
      const next = n.path.shift();
      const dx = next.x - n.x;
      const dy = next.y - n.y;
      n.dir = dx > 0 ? "right" : dx < 0 ? "left" : dy > 0 ? "down" : "up";
      n.step = { fx: n.x, fy: n.y, tx: next.x, ty: next.y, t0: now, dur };
    }
  }
}

function enterBuilding(locId) {
  G.player.inside = locId;
  G.panelCursor = 0;
  renderPanel();
}

function leaveBuilding() {
  const p = G.player;
  const door = G.locById[p.inside].door;
  p.inside = null;
  p.x = door[0];
  p.y = door[1] + 1;
  p.dir = "down";
  renderPanel();
}

// ---------- rendering ----------
function camera() {
  const pp = lerpPos(G.player);
  return {
    camX: clamp(Math.round(pp.x * 16 + 8 - VIEW_W / 2), 0, G.map.width * 16 - VIEW_W),
    camY: clamp(Math.round(pp.y * 16 + 8 - VIEW_H / 2), 0, G.map.height * 16 - VIEW_H),
  };
}

function render(now) {
  const { camX, camY } = camera();
  ctx.clearRect(0, 0, VIEW_W, VIEW_H);
  drawLayers(ctx, G.map.below, camX, camY, now);
  const chars = [];
  for (const n of G.npcs.values()) {
    if (npcVisible(n)) chars.push({ img: G.sheets[G.people[n.id].sprite], pos: lerpPos(n), dir: n.dir, moving: !!n.step, n });
  }
  const p = G.player;
  if (!p.inside) chars.push({ img: G.sheets[G.playerState.sprite], pos: lerpPos(p), dir: p.dir, moving: !!p.step });
  chars.sort((a, b) => a.pos.y - b.pos.y);
  for (const c of chars) drawCharacter(ctx, c.img, c.pos.x, c.pos.y, c.dir, c.moving, now, camX, camY);
  drawLayers(ctx, G.map.above, camX, camY, now);
  drawArch(ctx, stagesDone(), camX, camY, now);
  for (const c of chars) {
    if (!c.n) continue;
    const bubble = G.alert === c.n.id ? G.bubbles.exclamation
      : c.n.rec.convo !== null && c.n.rec.tool_use === null ? G.bubbles.dots : null;
    if (bubble) ctx.drawImage(bubble, Math.round(c.pos.x * 16 - camX), Math.round(c.pos.y * 16 - 32 - camY));
  }
  renderLabels(camX, camY);
}

function renderLabels(camX, camY) {
  const p = G.player;
  const items = [];
  for (const loc of G.locations) {
    if (!loc.door) continue;
    const [dx, dy] = loc.door;
    const occ = [...G.npcs.values()].filter((n) => n.rec.loc === loc.id && n.rec.inside && !npcWalking(n));
    const near = !p.inside && Math.max(Math.abs(p.x - dx), Math.abs(p.y - dy)) <= PLATE_RANGE;
    if (near) items.push(["plate", loc.name, (dx * 16 + 8 - camX) * SCALE, (dy * 16 - 18 - camY) * SCALE]);
    const working = occ.find((n) => n.rec.tool_use !== null && G.toolUses.has(n.rec.tool_use));
    const badge = working ? `${occ.length} ${G.toolById[G.toolUses.get(working.rec.tool_use).tool].emoji}` : String(occ.length);
    if (occ.length) items.push(["badge", badge, (dx * 16 + 15 - camX) * SCALE, (dy * 16 - camY) * SCALE]);
  }
  for (const tool of G.tools) {
    if (!tool.tile) continue;
    const [tx, ty] = tool.tile;
    items.push(["toolmark", tool.emoji, (tx * 16 + 8 - camX) * SCALE, (ty * 16 - camY) * SCALE]);
  }
  const stageCount = G.project.stages.length;
  items.push(["badge", stagesDone() === stageCount ? "💝 done!" : `💝 ${stagesDone()}/${stageCount}`,
    (9 * 16 + 8 - camX) * SCALE, (27 * 16 - 10 - camY) * SCALE]);
  for (const n of G.npcs.values()) {
    if (!npcVisible(n) || n.rec.tool_use === null || !G.toolUses.has(n.rec.tool_use)) continue;
    const pos = lerpPos(n);
    const emoji = G.toolById[G.toolUses.get(n.rec.tool_use).tool].emoji;
    items.push(["toolmark npc", emoji, Math.round((pos.x * 16 + 8 - camX) * SCALE),
      Math.round((pos.y * 16 - 16 - camY) * SCALE)]);
  }
  const sig = JSON.stringify(items);
  if (sig === G.labelSig) return;
  G.labelSig = sig;
  $("#labels").replaceChildren(...items.map(([cls, text, x, y]) => {
    const node = ui.el("div", cls, text);
    node.style.left = `${x}px`;
    node.style.top = `${y}px`;
    return node;
  }));
}

function clockText() {
  const [h, m] = G.frame.time.split(":").map(Number);
  const total = h * 60 + m + Math.floor((G.tickElapsed / TICK_MS) * 10);
  const hh = String(Math.floor(total / 60) % 24).padStart(2, "0");
  const mm = String(total % 60).padStart(2, "0");
  return `TUE ${hh}:${mm} ×${speed()}`;
}

function updateHud() {
  const p = G.player;
  const loc = playerLoc();
  const ps = G.playerState;
  const party = ps.attended ? "attended" : ps.flags.includes("party_accepted") ? "going"
    : ps.knowledge.includes("party") ? "invited" : "?";
  const election = ps.knowledge.includes("election") ? "heard" : "?";
  ui.hud({
    clock: clockText(),
    area: p.inside ? G.locById[p.inside].name : loc ? G.locById[loc].name : "Reward Town",
    status: `🎉 Party: ${party}\n🗳 Election: ${election}\n🔧 Ways: ${waysKnown()}/${G.tools.length * 6}\n` +
      `💝 Arch: ${stagesDone()}/${G.project.stages.length}`,
  });
}

function waysKnown() {
  return Object.values(G.playerState.techniques).reduce((s, ids) => s + ids.length, 0);
}

// Rows of the building panel: occupants first, then the tools of the building.
function panelRows() {
  const inside = G.player.inside;
  return [
    ...[...G.npcs.values()].filter((n) => n.rec.loc === inside).map((n) => ({ kind: "npc", id: n.id })),
    ...G.tools.filter((t) => t.location === inside).map((t) => ({ kind: "tool", id: t.id })),
  ];
}

// The running NPC tool use of `toolId`, from the current frame's records (helper/quality from G.toolUses).
function activeUse(toolId) {
  for (const rec of Object.values(G.frame?.agents ?? {})) {
    const r = rec.tool_use !== null ? G.toolUses.get(rec.tool_use) : null;
    if (r && r.tool === toolId && r.until_t !== null) return r;
  }
  return null;
}

function renderPanel() {
  const panel = $("#panel");
  const inside = G.player.inside;
  if (!inside || G.phase !== "playing") {
    panel.classList.add("hidden");
    return;
  }
  const rows = panelRows();
  G.panelCursor = clamp(G.panelCursor, 0, Math.max(0, rows.length - 1));
  panel.replaceChildren(ui.el("h2", null, G.locById[inside].name));
  if (!rows.some((r) => r.kind === "npc")) panel.append(ui.el("div", "empty", "Nobody else is here."));
  rows.forEach((r, i) => {
    if (r.kind === "tool" && (i === 0 || rows[i - 1].kind !== "tool")) panel.append(ui.el("h3", null, "Tools"));
    const row = ui.el("div", `occ${r.kind === "tool" ? " tool" : ""}${i === G.panelCursor ? " cur" : ""}`);
    const thumb = ui.el("div", "thumb");
    let who;
    if (r.kind === "npc") {
      const n = G.npcs.get(r.id);
      frontPortrait(thumb, G.people[r.id].sprite, 0.5);
      who = ui.el("div", "who", G.people[r.id].name);
      who.append(ui.el("span", "act", `${n.rec.emoji} ${n.rec.action}`));
    } else {
      const tool = G.toolById[r.id];
      const use = activeUse(r.id);
      thumb.textContent = tool.emoji;
      who = ui.el("div", "who", tool.name);
      const status = !use ? "free"
        : use.helper !== null ? `${firstName(use.user)} + ${firstName(use.helper)} · Q${use.quality}`
          : `${firstName(use.user)} is using it · Q${use.quality}`;
      who.append(ui.el("span", "act", status));
    }
    row.append(thumb, who);
    if (r.kind === "npc" && G.alert === r.id) {
      const alert = ui.el("div", "alert");
      alert.style.backgroundImage = 'url("assets/tuxemon/ui/bubble_exclamation.png")';
      row.append(alert);
    }
    panel.append(row);
  });
  panel.append(ui.el("div", "hint", "↑↓ choose · A talk/use · B leave"));
  panel.classList.remove("hidden");
}

async function openTool(toolId) {
  if (G.busy || G.stepInFlight) return;
  G.busy = true;
  try {
    await toolActions(toolId, flowCtx());
  } catch (err) {
    ui.toast(`⚠ ${err.message}`);
  } finally {
    G.busy = false;
  }
}

async function openSite() {
  if (G.busy || G.stepInFlight) return;
  G.busy = true;
  try {
    await siteActions(flowCtx());
  } catch (err) {
    ui.toast(`⚠ ${err.message}`);
  } finally {
    G.busy = false;
  }
}

// ---------- flows ----------
async function talkTo(id, { face }) {
  if (G.busy || G.stepInFlight) return;
  G.busy = true;
  try {
    const n = G.npcs.get(id);
    const person = G.people[id];
    if (face) n.dir = OPPOSITE[G.player.dir];
    await ui.say(`${person.name} ${n.rec.action}.`);
    const opts = await api.talkOptions(id);
    if (!opts.available) {
      await ui.say(opts.reason);
      return;
    }
    for (;;) {
      const labels = [...opts.topics.map((t) => t.label), "What are you thinking?", "Bye"];
      const reasons = [...opts.topics.map((t) => (t.enabled ? null : t.reason)), null, null];
      const i = await ui.choose(labels, { disabledReasons: reasons, title: person.name });
      if (i === null || i === labels.length - 1) return;
      if (i === labels.length - 2) {
        await showMind(id);
        continue;
      }
      await processResponse(await api.talk(id, opts.topics[i].id));
      return;
    }
  } catch (err) {
    ui.toast(`⚠ ${err.message}`);
  } finally {
    G.busy = false;
  }
}

function interact() {
  const p = G.player;
  if (p.step) return;
  const tx = p.x + DELTA[p.dir][0];
  const ty = p.y + DELTA[p.dir][1];
  const loc = playerLoc();
  const tool = G.tools.find((t) => t.tile && t.tile[0] === tx && t.tile[1] === ty && t.location === loc);
  if (tool) {
    openTool(tool.id);
    return;
  }
  const onArch = ([x, y]) => (x === tx && y === ty) || (x === p.x && y === p.y);
  if (loc === G.project.location && G.project.tiles.some(onArch)) {
    openSite();
    return;
  }
  for (const n of G.npcs.values()) {
    if (!npcVisible(n)) continue;
    const pos = n.step ? { x: n.step.tx, y: n.step.ty } : n;
    if ((n.x === tx && n.y === ty) || (pos.x === tx && pos.y === ty)) {
      if (npcWalking(n)) {
        G.busy = true;
        ui.say(`${G.people[n.id].name} is in a hurry!`).finally(() => { G.busy = false; });
      } else {
        talkTo(n.id, { face: true });
      }
      return;
    }
  }
}

function panelInput(btn) {
  const rows = panelRows();
  if (btn === "up" && G.panelCursor > 0) G.panelCursor -= 1;
  else if (btn === "down" && G.panelCursor < rows.length - 1) G.panelCursor += 1;
  else if (btn === "a" && rows.length) {
    const r = rows[G.panelCursor];
    if (r.kind === "npc") talkTo(r.id, { face: false });
    else openTool(r.id);
  }
  else if (btn === "b") leaveBuilding();
  renderPanel();
}

async function showMind(id) {
  const name = G.people[id].name;
  const d = await api.mind(id);
  if (d.none) {
    await ui.say(`${name} hasn't made a decision yet today.`);
    return;
  }
  const order = d.candidates.map((c, i) => ({ ...c, i })).sort((a, b) => b.score - a.score);
  let showPrompt = false;
  let offset = 0;
  const render = () => {
    const sheet = ui.openSheet();
    sheet.append(ui.el("h2", null, `At ${hhmm(d.t)} the model scored ${order.length} options for ${name}:`));
    const list = ui.el("div", "list");
    if (showPrompt) {
      list.append(ui.el("pre", null, d.prompt));
    } else {
      for (const c of order.slice(offset)) {
        const chosen = c.i === d.chosen;
        list.append(ui.el("div", `item${chosen ? " chosen" : ""}`, `${chosen ? "▶ " : ""}${c.score.toFixed(2)}  ${c.text}`));
      }
    }
    sheet.append(list, ui.el("div", "hint", `A ${showPrompt ? "scores" : "prompt"} · ↑↓ scroll · B close`));
  };
  render();
  await new Promise((resolve) => {
    const pop = ui.pushInput((btn) => {
      if (btn === "b") {
        pop();
        ui.closeSheet();
        resolve();
        return;
      }
      if (btn === "a") showPrompt = !showPrompt;
      else if (btn === "up") offset = Math.max(0, offset - 1);
      else if (btn === "down") offset = Math.min(order.length - 1, offset + 1);
      render();
    });
  });
}

async function showLog() {
  G.busy = true;
  try {
    const convs = [...G.conversations].reverse();
    await ui.browse({
      title: "Conversation log",
      items: convs.map((c) => ({
        text: `${hhmm(c.t)} ${firstName(c.participants[0])} → ${firstName(c.participants[1])} · ` +
          `${TOPIC_LABELS[c.topic] ?? c.topic} · ${c.effect}`,
      })),
      hint: "A watch · B close",
      onA: (i) => watchConversation(convs[i], G.people),
    });
  } finally {
    G.busy = false;
  }
}

async function withBusy(fn) {
  G.busy = true;
  try {
    await fn();
  } catch (err) {
    ui.toast(`⚠ ${err.message}`);
  } finally {
    G.busy = false;
  }
}

async function showPause() {
  G.busy = true;
  try {
    const st = await api.state();
    const m = st.meta;
    const sheet = ui.openSheet();
    sheet.append(ui.el("h2", null, "PAUSED"));
    for (const line of [
      `Model: ${m.model}`,
      `Device: ${m.device} (${m.gpu}) · ${m.dtype}`,
      `Reward model calls: ${m.rm_stats.calls} · sequences: ${m.rm_stats.sequences} · ${m.rm_stats.seconds.toFixed(1)}s`,
      "",
      "Art: Tuxemon project (CC BY-SA 4.0 and others) — assets/tuxemon/ATTRIBUTIONS.md",
    ]) sheet.append(ui.el("div", "item muted", line));
    const i = await ui.choose(["Resume", "New game"]);
    ui.closeSheet();
    if (i === 1) {
      G.busy = false;
      await newGameFlow();
    }
  } catch (err) {
    ui.closeSheet();
    ui.toast(`⚠ ${err.message}`);
  } finally {
    G.busy = false;
  }
}

async function showSummary(summary) {
  G.phase = "ended";
  renderPanel();
  $("#title").classList.add("hidden"); // reached from the title menu too: the title overlay would cover the sheet
  const names = (ids) => (ids.length ? ids.map(firstName).join(", ") : "nobody");
  const sheet = ui.openSheet();
  sheet.append(ui.el("h2", null, "The day is over!"));
  const you = summary.player;
  const lines = [
    `🎉 Party attendees: ${names(summary.attended_party)}`,
    `🗳 Will vote for Sam: ${names(summary.election_support)}`,
    `🗳 Undecided: ${names(summary.election_undecided)}`,
    `💬 Conversations today: ${summary.conversations}`,
    `🔧 Tool uses: ${summary.tools.uses} · helped: ${summary.tools.helped} · learned: ${summary.tools.learned}`,
    ...Object.entries(summary.tools.best).map(([id, b]) => {
      const tool = G.toolById[id];
      return `   ${tool.emoji} best ${tool.product}: ${firstName(b.user)}${b.helper ? ` + ${firstName(b.helper)}` : ""} Q${b.quality}`;
    }),
    `💝 Heart arch: ${summary.build.stages_done}/${G.project.stages.length}` +
      `${summary.build.finished ? ` — finished at ${summary.build.finished_time}` : ""} · ` +
      `builders: ${names(summary.build.builders)} · avg Q${summary.build.quality ?? "–"}`,
    "",
  ];
  if (you) {
    lines.push(
      `You ${you.attended_party ? "attended" : "did not attend"} the party` +
        `${you.accepted_party || you.attended_party ? "." : you.knows_party ? " (you declined)." : " (nobody invited you)."}`,
      `You told: ${names(you.told_party)} · they accepted: ${names(you.told_party_accepted)}`,
      `You used tools ${you.tool_uses}× · helped ${you.helped}× · built ${you.built} arch stage(s)`,
      `Best quality ${you.best_quality ?? "–"} · taught ${you.taught} · ways known ${you.ways_known}/${G.tools.length * 6}`,
    );
  }
  for (const line of lines) sheet.append(ui.el("div", "item", line));
  sheet.append(ui.el("div", "hint", "A: title screen"));
  await ui.waitA();
  ui.closeSheet();
  showTitle();
}

// ---------- title & new game ----------
function showTitle() {
  G.phase = "title";
  $("#hud").classList.add("hidden");
  $("#panel").classList.add("hidden");
  const title = $("#title");
  title.classList.remove("hidden");
  title.replaceChildren(
    ui.el("h1", null, "REWARD TOWN"),
    ui.el("div", "sub", "Every word and deed is chosen by Skywork-Reward-V2-Qwen3-0.6B"),
    ui.el("div", "press", "Press A"),
    ui.el("div", "credit", "Art: Tuxemon project (CC BY-SA 4.0 and others)"),
  );
  const pop = ui.pushInput((btn) => {
    if (btn !== "a") return;
    pop();
    titleMenu().catch((err) => fatal(err));
  });
}

async function titleMenu() {
  const st = await api.state();
  if (st.phase === "playing") {
    const i = await ui.choose(["Continue", "New game"], { cancel: false });
    return i === 0 ? startWorld(st, false) : newGameFlow();
  }
  if (st.phase === "ended") {
    const i = await ui.choose(["Day summary", "New game"], { cancel: false });
    if (i === 0) {
      setPeople(st);
      return showSummary(st.summary);
    }
  }
  return newGameFlow();
}

async function newGameFlow() {
  G.phase = "intro";
  const title = $("#title");
  title.classList.remove("hidden");
  title.replaceChildren();
  $("#hud").classList.add("hidden");
  $("#panel").classList.add("hidden");
  await ui.say("Welcome to Reward Town!");
  await ui.say("It's Tuesday, February 14, 2023.");
  await ui.say("Everyone here decides what to do and say by letting a reward model score their options.");
  await ui.show("Are you a boy or a girl?");
  const avatar = (await ui.choose(["Boy", "Girl"], { cancel: false })) === 0 ? "boy" : "girl";
  ui.hideDialog();
  for (;;) {
    const name = await ui.askName("What's your name?");
    if (NAME_RE.test(name)) {
      await ui.say("Rumor has it something special is happening at Hobbs Cafe tonight...");
      ui.show("Waking up the town...");
      try {
        const st = await api.newGame(name, avatar);
        ui.hideDialog();
        startWorld(st, true);
        return;
      } catch (err) {
        await ui.say(`Oops: ${err.message}`);
      }
    } else {
      await ui.say("Names start with a letter and use only letters, spaces, . ' and -.");
    }
  }
}

function setPeople(st) {
  G.meta = st.meta;
  G.locations = st.locations;
  G.locById = Object.fromEntries(st.locations.map((l) => [l.id, l]));
  G.tools = st.tools;
  G.toolById = Object.fromEntries(st.tools.map((t) => [t.id, t]));
  G.project = st.project;
  G.people = Object.fromEntries(st.agents.map((a) => [a.id, { name: a.name, age: a.age, sprite: a.sprite }]));
  if (st.player) G.people[PLAYER_ID] = { name: st.player.name, age: 25, sprite: st.player.sprite };
}

function startWorld(st, fresh) {
  setPeople(st);
  G.playerState = st.player;
  G.conversations = [...st.conversations];
  G.toolUses = new Map(st.tool_uses.map((r) => [r.id, r]));
  G.revealed = new Set();
  initNpcs(st.frame);
  const p = G.player;
  p.step = null;
  p.inside = null;
  if (fresh || !st.frame.player || st.frame.player.x === null) {
    [p.x, p.y] = st.meta.player_start;
    p.dir = "up";
  } else {
    const fp = st.frame.player;
    p.x = fp.x;
    p.y = fp.y;
    p.dir = "down";
    if (fp.loc && G.locById[fp.loc].door) p.inside = fp.loc;
  }
  G.tickElapsed = 0;
  G.alert = null;
  G.labelSig = "";
  G.phase = "playing";
  $("#title").classList.add("hidden");
  renderPanel();
  if (st.pending) {
    G.busy = true;
    processResponse({ pending: st.pending, frame: null, new_conversations: [], tool_uses: [], player: st.player,
      ended: false })
      .catch((err) => ui.toast(`⚠ ${err.message}`))
      .finally(() => { G.busy = false; });
  }
}

// ---------- input & loop ----------
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === "Shift") G.shift = true;
  const btn = KEYS[e.code];
  if (!btn) return;
  e.preventDefault();
  if (DIRS.has(btn) && !G.held.includes(btn)) G.held.push(btn);
  if (e.repeat && !DIRS.has(btn)) return;
  if (ui.dispatch(btn)) return;
  if (G.phase !== "playing" || G.busy) return;
  if (btn === "speed") G.speedIdx = (G.speedIdx + 1) % SPEEDS.length;
  else if (btn === "log") showLog();
  else if (btn === "work") withBusy(() => showWorkLog(flowCtx()));
  else if (btn === "skills") withBusy(() => showSkills(flowCtx()));
  else if (btn === "pause") showPause();
  else if (G.player.inside) panelInput(btn);
  else if (btn === "a") interact();
});
window.addEventListener("keyup", (e) => {
  if (e.key === "Shift") G.shift = false;
  const btn = KEYS[e.code];
  if (DIRS.has(btn)) G.held = G.held.filter((d) => d !== btn);
});
window.addEventListener("blur", () => {
  G.held = [];
  G.shift = false;
});

let last = performance.now();
function loop(now) {
  const dt = Math.min(100, now - last);
  last = now;
  if (G.phase === "playing") {
    updatePlayer(now);
    updateNpcs(now);
    if (!ui.hasModal() && !G.busy && !G.stepInFlight) {
      G.tickElapsed += dt * speed();
      if (G.tickElapsed >= TICK_MS) doStep();
    }
    render(now);
    updateHud();
  }
  requestAnimationFrame(loop);
}

function fatal(err) {
  console.error(err);
  const title = $("#title");
  title.classList.remove("hidden");
  title.replaceChildren(ui.el("h1", null, "ERROR"), ui.el("div", "sub", err.message));
}

async function boot() {
  $("#title").replaceChildren(ui.el("h1", null, "REWARD TOWN"), ui.el("div", "sub", "Loading..."));
  const [map, st] = await Promise.all([loadMap(), api.state()]);
  G.map = map;
  const names = [...new Set([...st.agents.map((a) => a.sprite), ...PLAYER_SPRITES])];
  const [sheets, ...bubbles] = await Promise.all([
    loadSprites(names),
    loadImage("assets/tuxemon/ui/bubble_dots.png"),
    loadImage("assets/tuxemon/ui/bubble_exclamation.png"),
  ]);
  G.sheets = sheets;
  G.bubbles = { dots: bubbles[0], exclamation: bubbles[1] };
  requestAnimationFrame(loop);
  if (st.phase === "ended") {
    // Reloading after the day ended goes straight to the results.
    setPeople(st);
    $("#title").classList.add("hidden");
    await showSummary(st.summary);
  } else {
    showTitle();
  }
}

boot().catch(fatal);
