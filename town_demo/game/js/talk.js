// Conversations in the overworld dialog box: live (the player picks their lines blind, then the judge's scores
// are revealed) or watch (replay from the talk log).

import * as api from "./api.js";
import * as ui from "./ui.js";

export const PLAYER_ID = "player";
export const TOPIC_LABELS = {
  party_invite: "Party invite",
  party_word_of_mouth: "Party gossip",
  election: "Campaign",
  small_talk: "Small talk",
};

export function outcomeText(effect, a, b) {
  return {
    accept: `${b} will come to the party!`,
    decline: `${b} can't make it to the party.`,
    support: `${b} will vote for Sam!`,
    undecided: `${b} is still undecided.`,
    none: `${a} and ${b} had a nice chat.`,
  }[effect] ?? `${a} and ${b} finished talking.`;
}

const argmax = (cands) => cands.reduce((best, c, i) => (c.score > cands[best].score ? i : best), 0);
// Candidates are scored as "Name: text"; menus show just the text.
const stripSpeaker = (text) => text.slice(text.indexOf(": ") + 2);

async function sayLine(line, people) {
  const speaker = people[line.speaker];
  await ui.say(`${speaker.name}: "${line.text}"`, { sprite: speaker.sprite });
}

async function judgeLine(line) {
  const best = argmax(line.candidates);
  const score = line.candidates[line.chosen].score;
  if (line.chooser === "player") {
    await ui.say(`Judge: your line scored ${score.toFixed(2)} · its pick was #${best + 1}` +
      `${best === line.chosen ? " — same as yours!" : ""}`);
  } else {
    await ui.say(`Judge picked #${line.chosen + 1} of ${line.candidates.length} · score ${score.toFixed(2)}`);
  }
}

const sameConversation = (a, b) =>
  a.topic === b.topic && a.initiator === b.initiator && a.responder === b.responder && a.t === b.t;

// pending: a "line" PlayerTurn view. ctx: {people, onResponse(resp), revealed: Set<string>}.
// Plays the conversation to its end and returns the last /api/choose response.
export async function runTalk(pending, ctx) {
  const { people } = ctx;
  const c = pending.context;
  const npcId = c.initiator === PLAYER_ID ? c.responder : c.initiator;
  await ui.say(c.initiator !== PLAYER_ID ? `${people[npcId].name} wants to talk!` : `You walk up to ${people[npcId].name}.`);
  await ui.say(`Topic: ${TOPIC_LABELS[c.topic] ?? c.topic}`);
  let view = pending;
  let shown = 0;
  const showLines = async (lines) => {
    for (; shown < lines.length; shown++) {
      await sayLine(lines[shown], people);
      if (lines[shown].chooser === "player") await judgeLine(lines[shown]);
    }
  };
  for (;;) {
    await showLines(view.context.lines);
    await ui.show("What do you say?");
    const i = await ui.choose(view.options.map((o) => stripSpeaker(o.text)), { wide: true, cancel: false });
    ui.hideDialog();
    const resp = await api.choose(i);
    ctx.onResponse(resp);
    if (resp.pending?.kind === "line" && sameConversation(resp.pending.context, c)) {
      view = resp.pending;
      continue;
    }
    const conv = resp.new_conversations.find((x) => x.topic === c.topic && x.t === c.t &&
      x.participants[0] === c.initiator && x.participants[1] === c.responder);
    if (conv) {
      await showLines(conv.lines);
      await ui.say(outcomeText(conv.effect, people[c.initiator].name, people[c.responder].name));
      ctx.revealed.add(`c:${conv.id}`);
    }
    return resp;
  }
}

// Replays a finished conversation with the judge's pick and score after every line.
export async function watchConversation(conv, people) {
  const [a, b] = conv.participants;
  await ui.say(`${people[a].name} and ${people[b].name} are talking.`);
  await ui.say(`Topic: ${TOPIC_LABELS[conv.topic] ?? conv.topic}`);
  for (const line of conv.lines) {
    await sayLine(line, people);
    await judgeLine(line);
  }
  await ui.say(outcomeText(conv.effect, people[a].name, people[b].name));
}
