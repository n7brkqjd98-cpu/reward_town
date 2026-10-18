// Work tools: the player's tool menu, blind technique choices with the judge's scores revealed afterwards,
// help requests from NPCs, the heart-arch site menu, the work log (J) and the skills sheet (K).
//
// ctx: {people: id -> {name, sprite}, toolById: id -> tool, toolUses: Map<id, record>, revealed: Set<string>,
//       locById, project: {name, stages, ...}, playerState(): player view, onResponse(resp), process(resp)}

import * as api from "./api.js";
import * as ui from "./ui.js";
import { PLAYER_ID } from "./talk.js";

const hhmm = (t) => `${String(Math.floor((t * 10) / 60)).padStart(2, "0")}:${String((t * 10) % 60).padStart(2, "0")}`;
const techText = (tool, id) => tool.techniques.find((x) => x.id === id).text;
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const stars = (n) => `${"★".repeat(n)}${"☆".repeat(4 - n)}`;
const argmaxOf = (cands, allowed) => cands.reduce(
  (best, c, i) => ((allowed === undefined || allowed.includes(c.technique)) &&
    (best === null || c.score > cands[best].score) ? i : best), null);

function firstName(ctx, id) {
  return ctx.people[id]?.name.split(" ")[0] ?? id;
}

const stageOf = (ctx, id) => ctx.project.stages.find((s) => s.id === id);

// Task/product of a record: its arch stage's when it has one, else its tool's.
function jobOf(record, ctx) {
  const job = record.stage !== null ? stageOf(ctx, record.stage) : ctx.toolById[record.tool];
  return { task: job.task, product: job.product };
}

function stagesDone(ctx) {
  let done = 0;
  for (const r of ctx.toolUses.values()) if (r.stage !== null && r.stage_done_t !== null) done += 1;
  return done;
}

function avgStars(record) {
  if (!record.reactions.length) return null;
  return record.reactions.reduce((s, r) => s + r.stars, 0) / record.reactions.length;
}

// Scrollable text sheet: lines [{text, cls?}]; A toggles `alt` text when given; B closes.
function textSheet(title, lines, alt) {
  let offset = 0;
  let showAlt = false;
  const render = () => {
    const sheet = ui.openSheet();
    sheet.append(ui.el("h2", null, title));
    const list = ui.el("div", "list");
    if (showAlt) list.append(ui.el("pre", null, alt));
    else for (const line of lines.slice(offset)) list.append(ui.el("div", `item${line.cls ? ` ${line.cls}` : ""}`, line.text));
    const hint = `${alt ? `A ${showAlt ? "scores" : "prompt"} · ` : ""}↑↓ scroll · B close`;
    sheet.append(list, ui.el("div", "hint", hint));
  };
  render();
  return new Promise((resolve) => {
    const pop = ui.pushInput((btn) => {
      if (btn === "b") {
        pop();
        ui.closeSheet();
        resolve();
        return;
      }
      if (btn === "a" && alt) showAlt = !showAlt;
      else if (btn === "up") offset = Math.max(0, offset - 1);
      else if (btn === "down") offset = Math.min(Math.max(0, lines.length - 1), offset + 1);
      render();
    });
  });
}

// Reveals the judge's view of the player's part in a tool use. `from`: reactions/learned already shown.
export async function revealToolUse(record, ctx, from = { reactions: 0, learned: 0 }) {
  const tool = ctx.toolById[record.tool];
  const helper = record.helper === PLAYER_ID;
  const cands = helper ? record.helper_candidates : record.candidates;
  const chosen = helper ? record.helper_chosen : record.chosen;
  const known = helper ? record.helper_known : record.known;
  const me = ctx.people[PLAYER_ID];
  await ui.say(`${me.name} ${techText(tool, cands[chosen].technique)}.`, { sprite: me.sprite });
  await ui.say(`Quality ${record.quality}/100` +
    `${record.helper !== null ? ` (alone it was ${record.solo_quality})` : ""}`);
  const bestKnown = argmaxOf(cands, known);
  await ui.say(`The judge liked “${techText(tool, cands[bestKnown].technique)}” most of the ways you know` +
    `${bestKnown === chosen ? " — your pick!" : "."}`);
  if (!known.includes(cands[argmaxOf(cands)].technique)) {
    await ui.say("The judge knows a better way you haven't learned yet. Watch others to learn it!");
  }
  for (const r of record.reactions.slice(from.reactions)) {
    await ui.say(`${r.text} ${stars(r.stars)}`, { sprite: ctx.people[r.observer].sprite });
  }
  for (const e of record.learned.slice(from.learned)) {
    if (e.by === PLAYER_ID) await ui.say(`${ctx.people[e.agent].name} learned your way by watching!`);
  }
  if (record.stage_done_t !== null) {
    await ui.say(`✔ The ${stageOf(ctx, record.stage).name} is done! (${stagesDone(ctx)}/${ctx.project.stages.length})`);
  }
}

// pending: a "technique" PlayerTurn view. Returns the /api/choose response.
export async function runTechnique(pending, ctx) {
  const c = pending.context;
  const tool = ctx.toolById[c.tool];
  if (c.role === "user") {
    await ui.show(c.stage ? `${capitalize(stageOf(ctx, c.stage).name)} of the heart arch: how will you use the ${tool.name}?`
      : `How will you use the ${tool.name}?`);
  } else {
    const user = ctx.people[c.user];
    await ui.show(`${user.name} ${techText(tool, c.partner_technique)}. How will you help?`, { sprite: user.sprite });
  }
  const i = await ui.choose(pending.options.map((o) => capitalize(techText(tool, o.technique))),
    { wide: true, cancel: false });
  ui.hideDialog();
  const prev = ctx.toolUses.get(c.use_id);
  const from = { reactions: prev?.reactions.length ?? 0, learned: prev?.learned.length ?? 0 };
  ctx.revealed.add(`u:${c.use_id}`);
  const resp = await api.choose(i);
  ctx.onResponse(resp);
  const record = resp.tool_uses.find((r) => r.id === c.use_id) ?? ctx.toolUses.get(c.use_id);
  if (record) await revealToolUse(record, ctx, from);
  return resp;
}

// pending: a "help_request" PlayerTurn view. Returns the /api/choose response.
export async function runHelpRequest(pending, ctx) {
  const c = pending.context;
  const tool = ctx.toolById[c.tool];
  const user = ctx.people[c.user];
  await ui.show(`${user.name} asks you to help with the ${tool.name}!`, { sprite: user.sprite });
  const i = await ui.choose(pending.options.map((o) => o.text), { cancel: false });
  ui.hideDialog();
  const resp = await api.choose(i);
  ctx.onResponse(resp);
  if (i !== 0) await ui.say(`${user.name} works on it alone.`);
  return resp;
}

async function showWays(tool, ctx) {
  const known = ctx.playerState().techniques[tool.id] ?? [];
  await textSheet(`${tool.emoji} Your ways with the ${tool.name} (${known.length}/6)`,
    tool.techniques.filter((x) => known.includes(x.id)).map((x) => ({ text: `· ${capitalize(x.text)}` })));
}

// The tool menu (panel row or facing an outdoor tool). Resolves when the menu closes or the job is done.
export async function toolActions(toolId, ctx) {
  const tool = ctx.toolById[toolId];
  for (;;) {
    const info = await api.tools();
    const entry = info.tools.find((t) => t.id === toolId);
    if (!entry) {
      await ui.say(`The ${tool.name} is not here.`);
      return;
    }
    const available = entry.helpers.filter((h) => h.available);
    const rows = ["Use it", "Use it with a helper…"];
    const reasons = [entry.use_reason, entry.use_reason ?? (available.length ? null : "Nobody here can help.")];
    const actions = ["use", "helper"];
    if (entry.in_use_by) {
      rows.push(`Help ${ctx.people[entry.in_use_by].name}`);
      reasons.push(entry.help_reason);
      actions.push("help");
    }
    rows.push(`Your ways (${entry.known}/6)`, "Back");
    reasons.push(null, null);
    actions.push("ways", "back");
    const i = await ui.choose(rows, { title: `${tool.emoji} ${tool.name}`, disabledReasons: reasons });
    const action = i === null ? "back" : actions[i];
    if (action === "back") return;
    if (action === "ways") {
      await showWays(tool, ctx);
      continue;
    }
    if (action === "use") return ctx.process(await api.useTool(toolId, null));
    if (action === "help") return ctx.process(await api.help(entry.use_id));
    const names = [...entry.helpers.map((h) => h.name), "Back"];
    const j = await ui.choose(names, {
      title: "Ask who to help?", disabledReasons: [...entry.helpers.map((h) => h.reason), null],
    });
    if (j === null || j === names.length - 1) continue;
    const asked = entry.helpers[j];
    const resp = await api.useTool(toolId, asked.id);
    const accepted = resp.pending?.kind === "technique" && resp.pending.context.helper === asked.id;
    await ui.say(accepted ? `${asked.name}: Sure, I'd be happy to help!` : `${asked.name}: Sorry, I can't help right now.`,
      { sprite: ctx.people[asked.id].sprite });
    return ctx.process(resp);
  }
}

function buildProgress(site, ctx) {
  const name = (id) => ctx.people[id]?.name ?? id;
  const lines = ctx.project.stages.map((s, k) => {
    const tool = ctx.toolById[s.tool];
    if (k < site.done.length) {
      const r = ctx.toolUses.get(site.done[k]);
      return { text: `✔ ${capitalize(s.name)} — ${name(r.user)}${r.helper !== null ? ` + ${name(r.helper)}` : ""} · ` +
        `Q${r.quality}`, cls: "chosen" };
    }
    if (k === site.done.length && site.in_use_by) return { text: `▶ ${capitalize(s.name)} — ${name(site.in_use_by)} is working…` };
    return { text: `· ${capitalize(s.name)} — needs the ${tool.emoji} ${tool.name}`, cls: "muted" };
  });
  if (site.stage) lines.push({ text: `You know ${site.known}/6 ways with the ${ctx.toolById[site.stage.tool].name}` });
  return textSheet(`💝 ${ctx.project.name}`, lines);
}

// The heart-arch menu (facing or standing on the arch). Resolves when the menu closes or the job is done.
export async function siteActions(ctx) {
  const n = ctx.project.stages.length;
  for (;;) {
    const site = await api.site();
    const rows = [];
    const reasons = [];
    const actions = [];
    if (site.stage) {
      const tool = ctx.toolById[site.stage.tool];
      rows.push(`Work on the ${site.stage.name} (${tool.emoji} ${tool.name})`);
      reasons.push(site.build_reason);
      actions.push("build");
    }
    if (site.in_use_by) {
      rows.push(`Help ${ctx.people[site.in_use_by].name} with the ${site.stage.name}`);
      reasons.push(site.help_reason);
      actions.push("help");
    }
    rows.push("Build progress", "Back");
    reasons.push(null, null);
    actions.push("progress", "back");
    const i = await ui.choose(rows, { title: `💝 ${ctx.project.name} · ${site.stage_index}/${n}`, disabledReasons: reasons });
    const action = i === null ? "back" : actions[i];
    if (action === "back") return;
    if (action === "progress") {
      await buildProgress(site, ctx);
      continue;
    }
    if (action === "build") return ctx.process(await api.build());
    return ctx.process(await api.help(site.use_id));
  }
}

function showToolUse(record, ctx) {
  const tool = ctx.toolById[record.tool];
  const name = (id) => ctx.people[id]?.name ?? id;
  const who = (chooser) => (chooser === "player" ? "your pick" : "best known way, judge's pick");
  const lines = [
    { text: `${name(record.user)} ${techText(tool, record.technique)}. (${who(record.chooser)})` },
  ];
  if (record.helper !== null) {
    lines.push({ text: `${name(record.helper)} helped: ${techText(tool, record.helper_technique)}. ` +
      `(${who(record.helper_chooser)})` });
  }
  if (record.asked !== null && record.ask_accepted === false) lines.push({ text: `${name(record.asked)} was asked but said no.` });
  lines.push({ text: `Quality ${record.quality}/100${record.helper !== null ? ` (alone ${record.solo_quality})` : ""}` });
  const scoreBlock = (title, cands, chosen, known) => {
    lines.push({ text: title, cls: "muted" });
    cands.map((c, i) => ({ ...c, i })).sort((a, b) => b.score - a.score).forEach((c) => {
      const mark = c.i === chosen ? "▶ " : "";
      lines.push({
        text: `${mark}${c.score.toFixed(2)} ${techText(tool, c.technique)}${known.includes(c.technique) ? "" : " (not known)"}`,
        cls: c.i === chosen ? "chosen" : undefined,
      });
    });
  };
  scoreBlock("Judge scores for all 6 ways:", record.candidates, record.chosen, record.known);
  if (record.helper !== null) {
    scoreBlock(`Judge scores with ${name(record.helper)} helping:`, record.helper_candidates, record.helper_chosen,
      record.helper_known);
  }
  for (const r of record.reactions) lines.push({ text: `${stars(r.stars)} ${r.text}` });
  for (const e of record.learned) {
    lines.push({ text: `📖 ${name(e.agent)} learned “${techText(tool, e.technique)}” from ${name(e.by)}` });
  }
  const loc = ctx.locById[record.loc]?.name ?? record.loc;
  const title = record.stage !== null
    ? `${hhmm(record.t)} ${tool.emoji} ${capitalize(stageOf(ctx, record.stage).name)} of the ${ctx.project.name}`
    : `${hhmm(record.t)} ${tool.emoji} ${capitalize(tool.name)} at ${loc}`;
  return textSheet(title, lines, record.prompt);
}

export async function showWorkLog(ctx) {
  const records = [...ctx.toolUses.values()].sort((a, b) => b.id - a.id);
  await ui.browse({
    title: "Work log",
    items: records.map((r) => {
      const tool = ctx.toolById[r.tool];
      const avg = avgStars(r);
      return {
        text: `${r.stage !== null ? "🏗 " : ""}${hhmm(r.t)} ${tool.emoji} ${firstName(ctx, r.user)}` +
          `${r.helper !== null ? `+${firstName(ctx, r.helper)}` : ""}` +
          ` · ${jobOf(r, ctx).product} · Q${r.quality} · ${avg === null ? "–" : avg.toFixed(1)}★ · ${r.learned.length}📖`,
      };
    }),
    hint: "A details · B close",
    onA: (i) => showToolUse(records[i], ctx),
  });
}

export async function showSkills(ctx) {
  const known = ctx.playerState().techniques;
  const lines = [];
  for (const tool of Object.values(ctx.toolById)) {
    const mine = known[tool.id] ?? [];
    lines.push({ text: `${tool.emoji} ${capitalize(tool.name)} — you know ${mine.length}/6`, cls: "chosen" });
    for (const x of tool.techniques) if (mine.includes(x.id)) lines.push({ text: `  · ${capitalize(x.text)}` });
    const credit = new Map(); // actor -> [sum, count]
    for (const r of ctx.toolUses.values()) {
      if (r.tool !== tool.id) continue;
      for (const actor of [r.user, r.helper]) {
        if (actor === null) continue;
        const acc = credit.get(actor) ?? [0, 0];
        for (const reaction of r.reactions) {
          acc[0] += reaction.stars;
          acc[1] += 1;
        }
        credit.set(actor, acc);
      }
    }
    const top = [...credit.entries()].filter(([, [, n]]) => n > 0)
      .map(([actor, [sum, n]]) => [actor, sum / n]).sort((a, b) => b[1] - a[1]).slice(0, 3);
    lines.push({
      text: `  Best in town: ${top.length ? top.map(([a, avg]) => `${firstName(ctx, a)} ${avg.toFixed(1)}★`).join(", ") : "nobody yet"}`,
      cls: "muted",
    });
  }
  await textSheet("Your skills", lines);
}
