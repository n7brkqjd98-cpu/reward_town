"""Reward Town engine: every NPC choice is the reward model's argmax over authored candidates.

The tick loop is a generator so a live game can pause for the player: `step()`, `player_talk()`,
`player_use_tool()` and `player_help()` yield a `PlayerTurn` whenever the player must pick (a conversation line, a
way to use a tool, or an answer to a help request), and resume with the chosen index.

Tools: NPCs (and the player) use physical work tools; the reward model scores all 6 techniques of a tool, the user
picks the best one it knows, and NPCs watching rate the result and may learn the techniques they saw.

Build: the Valentine's heart arch in Johnson Park is built in stages, each with its own tool. Nobody asks for help
on a stage: NPCs who have seen the arch pick up the next stage's tool themselves, and others join or watch.
"""

import math
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime

from . import world
from .scorer import RewardScorer

MAX_TICKS = 144
TICK_MINUTES = 10
DATE_LINE = "Today is Tuesday, February 14, 2023."
DECISION_QUESTION = ("What should {name} do next? Answer with the single most realistic next activity for "
                     "{name} right now.")
PLAYER_ID = "player"
NOT_READY = "You need a moment before your next job."


def hhmm(t: int) -> str:
    return f"{t * TICK_MINUTES // 60:02d}:{t * TICK_MINUTES % 60:02d}"


def wake_tick(persona: world.Persona) -> int:
    h, m = (int(part) for part in persona.wake.split(":"))
    return (h * 60 + m) // TICK_MINUTES


def quality(score: float) -> int:
    return round(100 / (1 + math.exp(-(score - world.TOOL_SCORE_OFFSET))))


def make_player(name: str, avatar: str) -> world.Persona:
    initials = "".join(word[0].upper() for word in name.split()[:2])
    return world.Persona(
        PLAYER_ID, name, initials, 25, "#ffffff", "", "07:00",
        "a newcomer who moved to town this week and is eager to meet the neighbors.", "",
        world.PLAYER_SPRITES[avatar],
    )


def initial_techniques(agent_id: str) -> dict[str, set[str]]:
    extra = world.INITIAL_TECHNIQUES.get(agent_id, {})
    return {tool.id: {f"{tool.id}_{i}" for i in range(world.BASIC_TECHNIQUES)}
            | {f"{tool.id}_{i}" for i in extra.get(tool.id, ())}
            for tool in world.ALL_TOOLS}


@dataclass
class Agent:
    persona: world.Persona
    loc: str | None  # None only for the player while outdoors outside the park
    action_id: str
    action_text: str
    emoji: str
    remaining: int
    counts: dict[str, int] = field(default_factory=dict)
    knowledge: set[str] = field(default_factory=set)
    flags: set[str] = field(default_factory=set)
    events: list[tuple[int, str]] = field(default_factory=list)
    engaged_tick: int | None = None  # engaged by someone else (conversation or help) during this tick
    techniques: dict[str, set[str]] = field(default_factory=dict)  # tool id -> known technique ids
    ratings: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # (actor id, tool id) -> stars given
    seen_project: bool = False  # has seen (or started) the heart arch, so the build option is offered

    @property
    def id(self) -> str:
        return self.persona.id

    @property
    def name(self) -> str:
        return self.persona.name


@dataclass(frozen=True)
class PlayerTurn:
    kind: str  # "line" | "technique" | "help_request"
    context: dict
    options: list[dict]  # line: {"text","score"}; technique: {"technique","text","score"}; help_request: {"text"}
    model_pick: int | None  # None for help_request


def persona_line(agent: Agent) -> str:
    p = agent.persona
    return f"{p.name} is {p.age}, {p.description}"


def knows_line(agent: Agent) -> str | None:
    parts = [text for key, text in world.KNOWLEDGE.items() if key in agent.knowledge]
    parts += [text.format(name=agent.name) for flag, text in world.FLAG_TEXT.items() if flag in agent.flags]
    return f"What {agent.name} knows: {' '.join(parts)}" if parts else None


def ratings_line(agent: Agent, name_of) -> str | None:
    if not agent.ratings:
        return None
    return (f"Work {agent.name} has watched today: "
            + "; ".join(f"{name_of(actor)}'s {world.TOOL_BY_ID[tool].product}: {sum(s) / len(s):.1f}/4 stars"
                        for (actor, tool), s in agent.ratings.items()) + ".")


def decision_prompt(agent: Agent, agents: list[Agent], t: int, player: Agent | None = None,
                    build: str | None = None) -> str:
    p = agent.persona
    others = [o.name for o in agents if o is not agent and o.loc == agent.loc]
    if player is not None and player.loc == agent.loc:
        others.append(player.name)
    here = f"Also here: {', '.join(others)}." if others else "Nobody else is here."
    done = ("; ".join(f"{hhmm(et)} {text}" for et, text in agent.events[-6:])
            or "Nothing yet; the day just started.")
    lines = [
        persona_line(agent),
        f"Lifestyle: {p.lifestyle}",
        f"{DATE_LINE} The time is {hhmm(t)}.",
        f"{p.name} is at {world.LOC_BY_ID[agent.loc].name}. {here}",
        f"What {p.name} has done today: {done}",
    ]
    knows = knows_line(agent)
    if knows:
        lines.append(knows)
    names = {a.id: a.name for a in agents}
    if player is not None:
        names[player.id] = player.name
    ratings = ratings_line(agent, names.__getitem__)
    if ratings:
        lines.append(ratings)
    if build is not None:
        lines.append(build)
    lines.append(DECISION_QUESTION.format(name=p.name))
    return "\n".join(lines)


def topic_allowed(topic: world.Topic, a: Agent, b: Agent, small_talk_done: bool) -> bool:
    if topic.initiators is not None and a.id not in topic.initiators:
        return False
    if a.id in topic.excluded_initiators:
        return False
    if topic.requires_flag and topic.requires_flag not in a.flags:
        return False
    if topic.requires_knowledge and topic.requires_knowledge not in a.knowledge:
        return False
    if topic.responder_lacks and topic.responder_lacks in b.knowledge:
        return False
    if topic.id == "small_talk" and small_talk_done:
        return False
    return True


def check_choice(choice, n: int) -> int:
    if not isinstance(choice, int) or not 0 <= choice < n:
        raise ValueError(f"choice must be an int in [0, {n}), got {choice!r}")
    return choice


def technique(tool: world.Tool, tech_id: str) -> world.Technique:
    return tool.techniques[int(tech_id.rsplit("_", 1)[1])]


def pick(cands: list[dict], known) -> int:
    """Index of the best-scoring candidate whose technique is in `known` (first wins ties)."""
    best = None
    for i, c in enumerate(cands):
        if c["technique"] in known and (best is None or c["score"] > cands[best]["score"]):
            best = i
    return best


def tool_dict(tool: world.Tool) -> dict:
    return {"id": tool.id, "name": tool.name, "emoji": tool.emoji, "location": tool.location,
            "tile": list(tool.tile) if tool.tile is not None else None, "task": tool.task,
            "product": tool.product, "techniques": [{"id": tech.id, "text": tech.text} for tech in tool.techniques]}


def stage_dict(stage: world.Stage) -> dict:
    return {"id": stage.id, "name": stage.name, "tool": stage.tool, "task": stage.task, "product": stage.product,
            "text": stage.text, "duration": stage.duration}


def project_dict() -> dict:
    P = world.PROJECT
    return {"id": P.id, "name": P.name, "location": P.location, "tiles": [list(tile) for tile in P.tiles],
            "hours": list(P.hours), "stages": [stage_dict(s) for s in P.stages]}


class Simulation:
    def __init__(self, scorer: RewardScorer, num_ticks: int = MAX_TICKS, player: world.Persona | None = None):
        self.scorer = scorer
        self.num_ticks = num_ticks
        self.agents = [
            Agent(
                persona=p, loc=p.home, action_id="sleep", action_text=world.ACTION_BY_ID["sleep"].text,
                emoji=world.ACTION_BY_ID["sleep"].emoji, remaining=wake_tick(p),
                knowledge=set(world.INITIAL_KNOWLEDGE.get(p.id, ())),
                flags=set(world.INITIAL_FLAGS.get(p.id, ())),
                techniques=initial_techniques(p.id), seen_project=p.id in world.PROJECT_STARTERS,
            )
            for p in world.PERSONAS
        ]
        self.agent_by_id = {a.id: a for a in self.agents}
        self.player = (None if player is None else
                       Agent(persona=player, loc=None, action_id="player", action_text="", emoji="", remaining=0,
                             techniques=initial_techniques(player.id)))
        self.player_pos: tuple[int, int] | None = None
        self.player_attended = False
        self.t = 0  # next tick to run
        self.ticks: list[dict] = []
        self.decisions: list[dict] = []
        self.conversations: list[dict] = []
        self.pair_counts: dict[frozenset[str], int] = {}
        self.small_talk_pairs: set[frozenset[str]] = set()
        self.tool_uses: list[dict] = []
        self.tool_events: list[int] = []  # tool-use ids, appended on creation and on every later change
        self.active_uses: dict[str, int] = {}  # NPC user id -> tool-use id
        self.active_helps: dict[str, int] = {}  # NPC helper id -> tool-use id
        self.reputation: dict[tuple[str, str], list[int]] = {}  # (actor id, tool id) -> stars from observers
        self.player_tool_t: int | None = None  # self.t when the player last used or helped with a tool
        self.project_done: list[int] = []  # tool-use ids of completed stages, in stage order
        self.project_finished_t: int | None = None

    def _agent(self, agent_id: str) -> Agent:
        return self.player if agent_id == PLAYER_ID else self.agent_by_id[agent_id]

    def _name(self, agent_id: str) -> str:
        return self._agent(agent_id).name

    def _score(self, prompt: str, responses: list[str]) -> tuple[list[dict], int]:
        scores = self.scorer.score(prompt, responses)
        chosen = scores.index(max(scores))
        return [{"text": r, "score": round(s, 4)} for r, s in zip(responses, scores)], chosen

    def _record(self, agent: Agent, decision: int | None, convo: int | None) -> dict:
        x, y = world.spot(agent.id, agent.loc)
        return {"loc": agent.loc, "x": x, "y": y, "inside": world.LOC_BY_ID[agent.loc].door is not None,
                "action": agent.action_text, "emoji": agent.emoji, "decision": decision, "convo": convo,
                "tool_use": self.active_uses.get(agent.id, self.active_helps.get(agent.id))}

    def _tool_busy(self, tool_id: str) -> bool:
        return any(self.tool_uses[rid]["tool"] == tool_id for rid in self.active_uses.values())

    # ---- heart arch ------------------------------------------------------------------------------------------
    def _current_stage(self) -> world.Stage | None:
        stages = world.PROJECT.stages
        return stages[len(self.project_done)] if len(self.project_done) < len(stages) else None

    def _active_build(self) -> int | None:
        return next((rid for rid in self.active_uses.values()
                     if self.tool_uses[rid]["project"] == world.PROJECT.id), None)

    def _job(self, record: dict) -> tuple[str, str]:
        if record["stage"] is not None:
            stage = world.STAGE_BY_ID[record["stage"]]
            return stage.task, stage.product
        tool = world.TOOL_BY_ID[record["tool"]]
        return tool.task, tool.product

    def _builders(self, record: dict) -> str:
        return self._name(record["user"]) + (f" and {self._name(record['helper'])}"
                                             if record["helper"] is not None else "")

    def build_line(self) -> str:
        P = world.PROJECT
        stage = self._current_stage()
        if stage is None:
            return f"The {P.name} in Johnson Park is finished."
        done = "; ".join(f"the {world.STAGE_BY_ID[self.tool_uses[rid]['stage']].name} by "
                         f"{self._builders(self.tool_uses[rid])}" for rid in self.project_done) or "nothing yet"
        active = self._active_build()
        working = (f" ({self._name(self.tool_uses[active]['user'])} is working on it now)"
                   if active is not None else "")
        return (f"The {P.name} in Johnson Park is being built. Done so far: {done}. Next step: the {stage.name}, "
                f"which needs the {world.TOOL_BY_ID[stage.tool].name}{working}.")

    def _decide(self, agent: Agent, t: int) -> Generator[PlayerTurn, int, tuple[int, int | None]]:
        hour = t * TICK_MINUTES // 60
        loc_name = world.LOC_BY_ID[agent.loc].name
        # ("action", Action) | ("build", Stage) | ("talk", other Agent, Topic) | ("help", tool-use id)
        options: list[tuple] = []
        responses: list[str] = []
        for action in world.ACTIONS:
            if not (world.hour_eligible(action, hour)
                    and (action.requires_flag is None or action.requires_flag in agent.flags)):
                continue
            count = agent.counts.get(action.id, 0)
            if action.tool is None:
                if not world.action_allowed_for(action, agent.id) or count >= action.max_per_day:
                    continue
            else:
                specialist = world.action_allowed_for(action, agent.id)
                if not specialist and len(agent.techniques[action.tool]) < world.SOLO_SKILL:
                    continue
                limit = (action.max_per_day if specialist
                         else min(action.max_per_day, world.LEARNED_TOOL_MAX_PER_DAY))
                if count >= limit or self._tool_busy(action.tool):
                    continue
            target = world.LOC_BY_ID[world.resolve_location(action, agent.persona)].name
            options.append(("action", action))
            responses.append(f"{agent.name} {action.text} at {target}.")
        P = world.PROJECT
        stage = self._current_stage()
        if (agent.seen_project and P.hours[0] <= hour < P.hours[1] and stage is not None
                and self._active_build() is None and not self._tool_busy(stage.tool)
                and len(agent.techniques[stage.tool]) >= world.SOLO_SKILL
                and agent.counts.get("build", 0) < world.BUILD_MAX_PER_DAY):
            options.append(("build", stage))
            responses.append(f"{agent.name} {stage.text} at Johnson Park.")
        targets = [o for o in self.agents if o is not agent and o.action_id != "sleep"]
        if self.player is not None and self.player.loc is not None:
            targets.append(self.player)
        for other in targets:
            if other.loc != agent.loc or other.engaged_tick == t:
                continue
            pair = frozenset((agent.id, other.id))
            if self.pair_counts.get(pair, 0) >= 2:
                continue
            for topic in world.TOPICS:
                if topic_allowed(topic, agent, other, pair in self.small_talk_pairs):
                    options.append(("talk", other, topic))
                    responses.append(
                        f"{agent.name} {topic.candidate_text.format(b=other.name)} at {loc_name}.")
        for rid in self.active_uses.values():
            record = self.tool_uses[rid]
            if record["loc"] == agent.loc and record["helper"] is None and record["user"] != agent.id:
                options.append(("help", rid))
                if record["stage"] is not None:
                    stage_name = world.STAGE_BY_ID[record["stage"]].name
                    responses.append(f"{agent.name} helps {self._name(record['user'])} with the {stage_name} of "
                                     f"the {world.PROJECT.name} at {loc_name}.")
                else:
                    tool = world.TOOL_BY_ID[record["tool"]]
                    responses.append(f"{agent.name} helps {self._name(record['user'])} with the {tool.name} at "
                                     f"{loc_name}.")

        prompt = decision_prompt(agent, self.agents, t, self.player,
                                 self.build_line() if agent.seen_project else None)
        candidates, chosen = self._score(prompt, responses)
        decision_id = len(self.decisions)
        self.decisions.append({"id": decision_id, "t": t, "agent": agent.id, "prompt": prompt,
                               "candidates": candidates, "chosen": chosen})

        option = options[chosen]
        if option[0] == "action":
            action = option[1]
            agent.loc = world.resolve_location(action, agent.persona)
            agent.action_id, agent.action_text, agent.emoji = action.id, action.text, action.emoji
            agent.remaining = action.duration
            agent.counts[action.id] = agent.counts.get(action.id, 0) + 1
            agent.events.append((t, f"{action.text} at {world.LOC_BY_ID[agent.loc].name}"))
            if action.tool is not None:
                yield from self._npc_tool_use(agent, action, t)
            return decision_id, None
        if option[0] == "build":
            stage = option[1]
            tool = world.TOOL_BY_ID[stage.tool]
            agent.loc = world.PROJECT.location
            agent.action_id, agent.action_text, agent.emoji = "build", stage.text, tool.emoji
            agent.remaining = stage.duration
            agent.counts["build"] = agent.counts.get("build", 0) + 1
            agent.events.append((t, f"{stage.text} at Johnson Park"))
            self._npc_build(agent, stage, t)
            return decision_id, None
        if option[0] == "help":
            self._join_use(agent, self.tool_uses[option[1]], t)
            return decision_id, None
        _, other, topic = option
        convo_id = yield from self._converse(agent, other, topic, t, live_tick=True)
        return decision_id, convo_id

    def _converse(self, a: Agent, b: Agent, topic: world.Topic, t: int, *,
                  live_tick: bool) -> Generator[PlayerTurn, int, int]:
        """Plays one 4-turn conversation. `live_tick`: started by an NPC decision inside tick t (updates tick
        records, pair limits and activity timers); False for player-initiated talks between ticks."""
        loc_name = world.LOC_BY_ID[a.loc].name
        spoken: list[str] = []
        lines: list[dict] = []
        effect = None
        for k, pool in enumerate(topic.turns):
            speaker, listener = (a, b) if k % 2 == 0 else (b, a)
            sp, lp = speaker.persona, listener.persona
            prompt_lines = [
                persona_line(speaker),
                f"The time is {hhmm(t)} at {loc_name}. {sp.name} is talking with {lp.name}, who is {lp.age}, "
                f"{lp.description}",
            ]
            knows = knows_line(speaker)
            if knows:
                prompt_lines.append(knows)
            prompt_lines.append("Conversation so far:")
            prompt_lines.extend(spoken or ["(nobody has spoken yet)"])
            prompt_lines.append(f"What does {sp.name} say next?")
            prompt = "\n".join(prompt_lines)
            texts = [line.text.format(a=a.name, b=b.name) for line in pool]
            candidates, chosen = self._score(prompt, [f"{sp.name}: {text}" for text in texts])
            chooser = "model"
            if self.player is not None and speaker is self.player:
                choice = yield PlayerTurn(
                    "line",
                    {"topic": topic.id, "initiator": a.id, "responder": b.id, "t": t, "time": hhmm(t),
                     "loc": a.loc, "turn": k, "lines": list(lines)},
                    candidates, chosen,
                )
                chosen, chooser = check_choice(choice, len(candidates)), "player"
            spoken.append(f"{sp.name}: {texts[chosen]}")
            lines.append({"speaker": speaker.id, "text": texts[chosen], "prompt": prompt,
                          "candidates": candidates, "chosen": chosen, "chooser": chooser})
            if k == 3:
                effect = pool[chosen].effect

        knowledge, flags = topic.effects[effect]
        b.knowledge |= knowledge
        b.flags |= flags
        a.events.append((t, f"talked with {b.name} ({topic.id}: {effect})"))
        b.events.append((t, f"talked with {a.name} ({topic.id}: {effect})"))
        if live_tick:
            pair = frozenset((a.id, b.id))
            self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
            if topic.id == "small_talk":
                self.small_talk_pairs.add(pair)

        convo_id = len(self.conversations)
        self.conversations.append({"id": convo_id, "t": t, "loc": a.loc, "topic": topic.id,
                                   "participants": [a.id, b.id], "effect": effect, "lines": lines})
        if not live_tick:
            return convo_id

        # Only NPC decisions start live conversations, so `a` is always an NPC here.
        a.action_id, a.action_text, a.emoji = "chat", f"is chatting with {b.name}", "💬"
        a.remaining = 1
        a.engaged_tick = t
        b.engaged_tick = t
        if b is self.player:
            return convo_id
        records = self.ticks[t]["agents"]
        b_record = {**self._record(b, records.get(b.id, {}).get("decision"), convo_id),
                    "action": f"is chatting with {a.name}", "emoji": "💬"}
        if b.id in records:
            b.remaining += 1  # already decremented this tick: pause B's own activity
        records[b.id] = b_record
        return convo_id

    # ---- tools -------------------------------------------------------------------------------------------
    def _work_prompt(self, user: Agent, tool: world.Tool, t: int, task: str, product: str) -> str:
        return (f"{DATE_LINE} The time is {hhmm(t)} at {world.LOC_BY_ID[tool.location].name}. {user.name} is "
                f"using the {tool.name} to {task}.\n"
                f"Which way of using the {tool.name} gives the best {product}?")

    def _tool_candidates(self, prompt: str, tool: world.Tool, responses: list[str]) -> list[dict]:
        scores = self.scorer.score(prompt, responses)
        return [{"technique": tech.id, "text": r, "score": round(s, 4)}
                for tech, r, s in zip(tool.techniques, responses, scores)]

    def _solo_candidates(self, prompt: str, user: Agent, tool: world.Tool) -> list[dict]:
        return self._tool_candidates(prompt, tool, [f"{user.name} {tech.text}." for tech in tool.techniques])

    def _joint_candidates(self, prompt: str, user: Agent, user_tech: str, helper: Agent,
                          tool: world.Tool) -> list[dict]:
        utext = technique(tool, user_tech).text
        return self._tool_candidates(prompt, tool, [f"{user.name} {utext}, while {helper.name} {tech.text}."
                                                    for tech in tool.techniques])

    def _known(self, agent: Agent, tool: world.Tool) -> list[str]:
        return [tech.id for tech in tool.techniques if tech.id in agent.techniques[tool.id]]

    def _player_technique(self, cands: list[dict], tool: world.Tool,
                          context: dict) -> Generator[PlayerTurn, int, int]:
        """Blind technique choice among the player's known techniques; returns the candidate index."""
        indices = [i for i, c in enumerate(cands) if c["technique"] in self.player.techniques[tool.id]]
        model_pick = indices.index(pick(cands, self.player.techniques[tool.id]))
        choice = yield PlayerTurn("technique", context, [cands[i] for i in indices], model_pick)
        return indices[check_choice(choice, len(indices))]

    def _accepts(self, helper: Agent, user: Agent, tool: world.Tool, t: int) -> bool:
        lines = [
            persona_line(helper),
            f"{DATE_LINE} The time is {hhmm(t)} at {world.LOC_BY_ID[tool.location].name}. {user.name} asks "
            f"{helper.name} for help using the {tool.name} to {tool.task}.",
        ]
        ratings = ratings_line(helper, self._name)
        if ratings:
            lines.append(ratings)
        lines.append(f"What does {helper.name} answer?")
        _, chosen = self._score("\n".join(lines), [a.format(h=helper.name) for a in world.HELP_ANSWERS])
        return chosen == 0

    def _new_use(self, t: int, tool: world.Tool, user: Agent, prompt: str, cands: list[dict], known: list[str],
                 chosen: int, chooser: str, until_t: int | None) -> dict:
        q = quality(cands[chosen]["score"])
        return {"id": len(self.tool_uses), "t": t, "time": hhmm(t), "loc": tool.location, "tool": tool.id,
                "user": user.id, "helper": None, "asked": None, "ask_accepted": None,
                "technique": cands[chosen]["technique"], "prompt": prompt, "candidates": cands, "known": known,
                "chosen": chosen, "chooser": chooser,
                "helper_technique": None, "helper_candidates": None, "helper_known": None, "helper_chosen": None,
                "helper_chooser": None, "solo_quality": q, "quality": q, "until_t": until_t,
                "reactions": [], "learned": [], "project": None, "stage": None, "stage_done_t": None}

    def _set_helper(self, record: dict, helper: Agent, cands: list[dict], known: list[str], chosen: int,
                    chooser: str) -> None:
        record.update(helper=helper.id, helper_technique=cands[chosen]["technique"], helper_candidates=cands,
                      helper_known=known, helper_chosen=chosen, helper_chooser=chooser,
                      quality=quality(cands[chosen]["score"]))

    def _engage_helper(self, helper: Agent, user: Agent, tool: world.Tool, use_id: int, t: int,
                       duration: int) -> None:
        helper.action_id, helper.action_text, helper.emoji = "help_tool", f"helps {user.name} with the {tool.name}", \
            tool.emoji
        helper.remaining = duration - 1
        helper.engaged_tick = t
        helper.counts["help_tool"] = helper.counts.get("help_tool", 0) + 1
        helper.events.append((t, f"helped {user.name} with the {tool.name}"))
        self.active_helps[helper.id] = use_id
        records = self.ticks[t]["agents"]
        records[helper.id] = self._record(helper, records.get(helper.id, {}).get("decision"), None)

    def _npc_tool_use(self, user: Agent, action: world.Action, t: int) -> Generator[PlayerTurn, int, None]:
        tool = world.TOOL_BY_ID[action.tool]
        loc_name = world.LOC_BY_ID[tool.location].name
        available = [o for o in self.agents
                     if o is not user and o.loc == user.loc and o.action_id != "sleep" and o.engaged_tick != t
                     and o.id not in self.active_uses and o.id not in self.active_helps]
        p = self.player
        if p is not None and p.loc == user.loc and p.engaged_tick != t:
            available.append(p)
        asked = helper = None
        accepted = None
        if available:
            lines = [persona_line(user),
                     f"{DATE_LINE} The time is {hhmm(t)} at {loc_name}. {user.name} is about to use the "
                     f"{tool.name} to {tool.task}."]
            ratings = ratings_line(user, self._name)
            if ratings:
                lines.append(ratings)
            lines.append(f"Also here: {', '.join(o.name for o in available)}.")
            lines.append(f"Should {user.name} ask someone for help?")
            _, ask = self._score("\n".join(lines), [f"{user.name} works on it alone."]
                                 + [f"{user.name} asks {o.name} to help with the {tool.name}." for o in available])
            if ask > 0:
                asked = available[ask - 1]
                if asked is p:
                    choice = yield PlayerTurn(
                        "help_request",
                        {"tool": tool.id, "user": user.id, "loc": tool.location, "t": t, "time": hhmm(t)},
                        [{"text": s} for s in world.PLAYER_HELP_ANSWERS], None)
                    accepted = check_choice(choice, len(world.PLAYER_HELP_ANSWERS)) == 0
                else:
                    accepted = self._accepts(asked, user, tool, t)
                helper = asked if accepted else None

        prompt = self._work_prompt(user, tool, t, tool.task, tool.product)
        cands = self._solo_candidates(prompt, user, tool)
        known = self._known(user, tool)
        record = self._new_use(t, tool, user, prompt, cands, known, pick(cands, known), "model",
                               t + action.duration)
        record.update(asked=asked.id if asked is not None else None, ask_accepted=accepted)
        shown = [(user, record["technique"])]
        if helper is not None:
            hcands = self._joint_candidates(prompt, user, record["technique"], helper, tool)
            hknown = self._known(helper, tool)
            if helper is p:
                hchosen = yield from self._player_technique(hcands, tool, {
                    "use_id": len(self.tool_uses), "tool": tool.id, "role": "helper", "user": user.id,
                    "helper": PLAYER_ID, "loc": tool.location, "t": t, "time": hhmm(t),
                    "partner_technique": record["technique"], "stage": None})
                hchooser = "player"
            else:
                hchosen, hchooser = pick(hcands, hknown), "model"
            self._set_helper(record, helper, hcands, hknown, hchosen, hchooser)
            shown.append((helper, record["helper_technique"]))

        self.tool_uses.append(record)
        self.tool_events.append(record["id"])
        self.active_uses[user.id] = record["id"]
        if helper is p and p is not None:
            p.engaged_tick = t
            p.events.append((t, f"helped {user.name} with the {tool.name}"))
        elif helper is not None:
            self._engage_helper(helper, user, tool, record["id"], t, action.duration)
        self._observe(record, t, shown)

    def _npc_build(self, user: Agent, stage: world.Stage, t: int) -> None:
        """An NPC starts the next arch stage on its own: no ask-for-help step, the build is wordless."""
        tool = world.TOOL_BY_ID[stage.tool]
        prompt = self._work_prompt(user, tool, t, stage.task, stage.product)
        cands = self._solo_candidates(prompt, user, tool)
        known = self._known(user, tool)
        record = self._new_use(t, tool, user, prompt, cands, known, pick(cands, known), "model", t + stage.duration)
        record.update(project=world.PROJECT.id, stage=stage.id)
        self.tool_uses.append(record)
        self.tool_events.append(record["id"])
        self.active_uses[user.id] = record["id"]
        self._observe(record, t, [(user, record["technique"])])

    def _finish_use(self, agent: Agent, t: int) -> None:
        rid = self.active_uses.pop(agent.id, None)
        self.active_helps.pop(agent.id, None)
        if rid is not None:
            self._complete_stage(self.tool_uses[rid], t)

    def _complete_stage(self, record: dict, t: int) -> None:
        if record["stage"] is None or record["stage_done_t"] is not None:
            return
        stage = world.STAGE_BY_ID[record["stage"]]
        record["stage_done_t"] = t
        self.project_done.append(record["id"])
        self.tool_events.append(record["id"])
        for agent_id in (record["user"], record["helper"]):
            if agent_id is not None:
                self._agent(agent_id).events.append((t, f"finished the {stage.name} of the {world.PROJECT.name}"))
        if len(self.project_done) == len(world.PROJECT.stages):
            self.project_finished_t = t

    def _join_use(self, helper: Agent, record: dict, t: int) -> None:
        """An NPC chose to join a running NPC tool use as its helper (inside its own decision in tick t)."""
        tool = world.TOOL_BY_ID[record["tool"]]
        user = self.agent_by_id[record["user"]]
        cands = self._joint_candidates(record["prompt"], user, record["technique"], helper, tool)
        known = self._known(helper, tool)
        self._set_helper(record, helper, cands, known, pick(cands, known), "model")
        self.tool_events.append(record["id"])
        self._engage_helper(helper, user, tool, record["id"], t, max(1, record["until_t"] - t))
        helper.remaining += 1  # step() decrements the deciding agent once more this tick
        self._observe(record, t, [(helper, record["helper_technique"])])

    def _observe(self, record: dict, t: int, shown: list[tuple[Agent, str]]) -> None:
        """Everyone at the tool's location watches: NPCs rate the result and may learn the techniques shown."""
        tool = world.TOOL_BY_ID[record["tool"]]
        loc_name = world.LOC_BY_ID[tool.location].name
        task, product = self._job(record)
        user, helper = record["user"], record["helper"]
        actors = self._name(user) + (f" and {self._name(helper)}" if helper is not None else "")
        result = (record["helper_candidates"][record["helper_chosen"]]["text"] if helper is not None
                  else record["candidates"][record["chosen"]]["text"])
        credited = [user] + ([helper] if helper is not None else [])
        for o in self.agents:
            if o.loc != record["loc"] or o.action_id == "sleep" or o.id in credited:
                continue
            prompt = "\n".join([
                persona_line(o),
                f"The time is {hhmm(t)} at {loc_name}. {o.name} watches {actors} use the {tool.name} to "
                f"{task}: {result}",
                f"What does {o.name} think of the {product}?",
            ])
            cands, k = self._score(prompt, [r.format(o=o.name, product=product) for r in world.REACTIONS])
            stars = 4 - k
            for actor in credited:
                o.ratings.setdefault((actor, tool.id), []).append(stars)
                self.reputation.setdefault((actor, tool.id), []).append(stars)
            record["reactions"].append({"observer": o.id, "stars": stars, "text": cands[k]["text"],
                                        "candidates": cands, "chosen": k})
            for actor, tech_id in shown:
                if tech_id in o.techniques[tool.id]:
                    continue
                tech = technique(tool, tech_id)
                known = "; ".join(x.text for x in tool.techniques if x.id in o.techniques[tool.id])
                prompt = "\n".join([
                    persona_line(o),
                    f"The time is {hhmm(t)} at {loc_name}. {o.name} just watched {actor.name} use the {tool.name}: "
                    f"{actor.name} {tech.text}.",
                    f"{o.name} rated the result {stars} out of 4 stars.",
                    f"{o.name} already knows these ways to use the {tool.name}: {known}.",
                    f"What does {o.name} do?",
                ])
                _, learn = self._score(prompt, [line.format(o=o.name, actor=actor.name, technique=tech.text,
                                                            tool=tool.name) for line in world.LEARN_LINES])
                if learn == 0:
                    o.techniques[tool.id].add(tech_id)
                    record["learned"].append({"agent": o.id, "technique": tech_id, "by": actor.id})
                    o.events.append((t, f"learned a new way to use the {tool.name} by watching {actor.name}"))
        p = self.player
        if p is not None and p.loc == record["loc"] and PLAYER_ID not in credited:
            for actor, tech_id in shown:
                if tech_id not in p.techniques[tool.id]:
                    p.techniques[tool.id].add(tech_id)
                    record["learned"].append({"agent": PLAYER_ID, "technique": tech_id, "by": actor.id})

    def player_tools(self) -> dict:
        p = self.player
        ready = self.player_tool_t != self.t
        here = [o for o in self.agents if p.loc is not None and o.loc == p.loc]
        helpers = []
        for o in here:
            busy = self.active_uses.get(o.id, self.active_helps.get(o.id))
            reason = (f"{o.name} is sleeping." if o.action_id == "sleep" else
                      f"{o.name} is busy with the {world.TOOL_BY_ID[self.tool_uses[busy]['tool']].name}."
                      if busy is not None else None)
            helpers.append({"id": o.id, "name": o.name, "available": reason is None, "reason": reason})
        tools = []
        for tool in world.TOOLS:
            if p.loc is None or tool.location != p.loc:
                continue
            use_id = next((rid for rid in self.active_uses.values() if self.tool_uses[rid]["tool"] == tool.id),
                          None)
            record = self.tool_uses[use_id] if use_id is not None else None
            use_reason = (f"{self._name(record['user'])} is using it. Help instead!" if record is not None else
                          NOT_READY if not ready else None)
            help_reason = ("Nobody is using it." if record is None else
                           f"{self._name(record['helper'])} is already helping." if record["helper"] is not None
                           else NOT_READY if not ready else None)
            tools.append({"id": tool.id, "use_id": use_id, "in_use_by": record["user"] if record else None,
                          "helper": record["helper"] if record else None,
                          "can_use": use_reason is None, "use_reason": use_reason,
                          "can_help": help_reason is None, "help_reason": help_reason,
                          "known": len(p.techniques[tool.id]), "helpers": helpers})
        return {"loc": p.loc, "ready": ready, "tools": tools}

    def player_site(self) -> dict:
        """The heart arch as seen by the player: the next stage and whether they can build it or help."""
        P, p = world.PROJECT, self.player
        here = p.loc == P.location
        ready = self.player_tool_t != self.t
        stage = self._current_stage()
        tool = world.TOOL_BY_ID[stage.tool] if stage is not None else None
        known = len(p.techniques[tool.id]) if tool is not None else None
        use_id = self._active_build()
        record = self.tool_uses[use_id] if use_id is not None else None
        hour = (self.t - 1) * TICK_MINUTES // 60
        away = f"Go to Johnson Park to work on the {P.name}."
        build_reason = (away if not here else
                        f"The {P.name} is finished!" if stage is None else
                        f"{self._name(record['user'])} is working on the {stage.name}. Help instead!"
                        if record is not None else
                        f"Building happens between {P.hours[0]:02d}:00 and {P.hours[1]:02d}:00."
                        if not P.hours[0] <= hour < P.hours[1] else
                        f"You need to know {world.SOLO_SKILL} ways to use the {tool.name} first. Watch others build "
                        f"to learn!" if known < world.SOLO_SKILL else
                        f"Someone is using the {tool.name}." if self._tool_busy(tool.id) else
                        NOT_READY if not ready else None)
        help_reason = (away if not here else
                       "Nobody is building right now." if record is None else
                       f"{self._name(record['helper'])} is already helping." if record["helper"] is not None else
                       NOT_READY if not ready else None)
        return {"here": here, "ready": ready, "stage": stage_dict(stage) if stage is not None else None,
                "stage_index": len(self.project_done), "done": list(self.project_done), "use_id": use_id,
                "in_use_by": record["user"] if record is not None else None,
                "helper": record["helper"] if record is not None else None,
                "can_build": build_reason is None, "build_reason": build_reason,
                "can_help": help_reason is None, "help_reason": help_reason, "known": known}

    def player_build(self) -> Generator[PlayerTurn, int, None]:
        """The player builds the next arch stage between ticks (timed at the last completed tick); it is instant."""
        t = self.t - 1
        self.player_tool_t = self.t
        p, stage = self.player, self._current_stage()
        tool = world.TOOL_BY_ID[stage.tool]
        prompt = self._work_prompt(p, tool, t, stage.task, stage.product)
        cands = self._solo_candidates(prompt, p, tool)
        known = self._known(p, tool)
        chosen = yield from self._player_technique(cands, tool, {
            "use_id": len(self.tool_uses), "tool": tool.id, "role": "user", "user": PLAYER_ID, "helper": None,
            "loc": tool.location, "t": t, "time": hhmm(t), "partner_technique": None, "stage": stage.id})
        record = self._new_use(t, tool, p, prompt, cands, known, chosen, "player", None)
        record.update(project=world.PROJECT.id, stage=stage.id)
        self.tool_uses.append(record)
        self.tool_events.append(record["id"])
        p.events.append((t, f"{stage.text} at Johnson Park"))
        self._observe(record, t, [(p, record["technique"])])
        self._complete_stage(record, t)

    def player_use_tool(self, tool_id: str, helper_id: str | None) -> Generator[PlayerTurn, int, None]:
        """The player uses a tool between ticks (timed at the last completed tick), optionally asking an NPC."""
        t = self.t - 1
        self.player_tool_t = self.t
        p, tool = self.player, world.TOOL_BY_ID[tool_id]
        asked = self.agent_by_id[helper_id] if helper_id is not None else None
        accepted = self._accepts(asked, p, tool, t) if asked is not None else None
        helper = asked if accepted else None
        prompt = self._work_prompt(p, tool, t, tool.task, tool.product)
        cands = self._solo_candidates(prompt, p, tool)
        known = self._known(p, tool)
        chosen = yield from self._player_technique(cands, tool, {
            "use_id": len(self.tool_uses), "tool": tool.id, "role": "user", "user": PLAYER_ID,
            "helper": helper.id if helper is not None else None, "loc": tool.location, "t": t, "time": hhmm(t),
            "partner_technique": None, "stage": None})
        record = self._new_use(t, tool, p, prompt, cands, known, chosen, "player", None)
        record.update(asked=helper_id, ask_accepted=accepted)
        shown = [(p, record["technique"])]
        p.events.append((t, f"used the {tool.name} at {world.LOC_BY_ID[tool.location].name}"))
        if helper is not None:
            hcands = self._joint_candidates(prompt, p, record["technique"], helper, tool)
            hknown = self._known(helper, tool)
            self._set_helper(record, helper, hcands, hknown, pick(hcands, hknown), "model")
            shown.append((helper, record["helper_technique"]))
            helper.events.append((t, f"helped {p.name} with the {tool.name}"))
        self.tool_uses.append(record)
        self.tool_events.append(record["id"])
        self._observe(record, t, shown)

    def player_help(self, use_id: int) -> Generator[PlayerTurn, int, None]:
        """The player joins a running NPC tool use as its helper, timed at the last completed tick."""
        t = self.t - 1
        self.player_tool_t = self.t
        p, record = self.player, self.tool_uses[use_id]
        tool, user = world.TOOL_BY_ID[record["tool"]], self.agent_by_id[record["user"]]
        cands = self._joint_candidates(record["prompt"], user, record["technique"], p, tool)
        known = self._known(p, tool)
        chosen = yield from self._player_technique(cands, tool, {
            "use_id": use_id, "tool": tool.id, "role": "helper", "user": user.id, "helper": PLAYER_ID,
            "loc": tool.location, "t": t, "time": hhmm(t), "partner_technique": record["technique"],
            "stage": record["stage"]})
        self._set_helper(record, p, cands, known, chosen, "player")
        self.tool_events.append(use_id)
        p.events.append((t, f"helped {user.name} with the {tool.name}"))
        self._observe(record, t, [(p, record["helper_technique"])])

    # ---- tick loop -----------------------------------------------------------------------------------------
    def step(self) -> Generator[PlayerTurn, int, None]:
        """Runs tick `self.t`; yields whenever the player must choose (a line, a technique, a help answer)."""
        t = self.t
        if t >= self.num_ticks:
            raise RuntimeError(f"day is over: all {self.num_ticks} ticks have run")
        records: dict[str, dict] = {}
        self.ticks.append({"t": t, "time": hhmm(t), "agents": records})
        for agent in self.agents:
            if agent.engaged_tick == t:
                continue
            decision = convo = None
            if agent.remaining <= 0:
                self._finish_use(agent, t)
                decision, convo = yield from self._decide(agent, t)
            records[agent.id] = self._record(agent, decision, convo)
            agent.remaining -= 1
        # keep agent order stable in the JSON
        self.ticks[t]["agents"] = {a.id: records[a.id] for a in self.agents}
        p = self.player
        if p is not None:
            x, y = self.player_pos if self.player_pos is not None else (None, None)
            self.ticks[t]["player"] = {"x": x, "y": y, "loc": p.loc}
            hour = t * TICK_MINUTES // 60
            if ("party_accepted" in p.flags and p.loc == "hobbs_cafe" and 17 <= hour < 19
                    and not self.player_attended):
                self.player_attended = True
                p.events.append((t, "attended the Valentine's Day party at Hobbs Cafe"))
        for a in self.agents:
            if a.loc == world.PROJECT.location and a.action_id != "sleep":
                a.seen_project = True
        self.t += 1

    def player_talk(self, npc_id: str, topic_id: str) -> Generator[PlayerTurn, int, None]:
        """Player-initiated conversation, timed at the last completed tick; exempt from pair limits."""
        yield from self._converse(self.player, self.agent_by_id[npc_id], world.TOPIC_BY_ID[topic_id],
                                  self.t - 1, live_tick=False)

    def player_topics(self, npc_id: str) -> dict:
        npc, p = self.agent_by_id[npc_id], self.player
        if npc.action_id == "sleep":
            return {"available": False, "reason": f"{npc.name} is sleeping.", "topics": []}
        if p.loc is None or npc.loc != p.loc:
            return {"available": False, "reason": f"{npc.name} is too far away.", "topics": []}
        invited = "party_accepted" in p.flags
        knows = "party" in npc.knowledge
        party_reason = (None if invited and not knows else
                        "You haven't been invited yet." if not invited else
                        f"{npc.name} already knows about the party.")
        return {"available": True, "reason": None, "topics": [
            {"id": "small_talk", "label": "Chat", "enabled": True, "reason": None},
            {"id": "party_word_of_mouth", "label": "Invite to the party", "enabled": party_reason is None,
             "reason": party_reason},
        ]}

    def run(self) -> None:
        """Headless day: no player input is possible."""
        for _ in range(self.num_ticks):
            t = self.t
            for _turn in self.step():
                raise RuntimeError("player input requested in a headless run")
            if t % 6 == 0:
                s = self.scorer.stats
                print(f"[{hhmm(t)}] decisions={len(self.decisions)} conversations={len(self.conversations)} "
                      f"tool_uses={len(self.tool_uses)} rm_calls={s['calls']} rm_s={s['seconds']:.2f}", flush=True)

    def summary(self) -> dict:
        ids = [a.id for a in self.agents]
        election = {c["participants"][1]: c["effect"] for c in self.conversations if c["topic"] == "election"}
        by_id = self.agent_by_id
        best: dict[str, dict] = {}
        for r in self.tool_uses:  # arch stages are reported under "build", not as a tool's best product
            if r["stage"] is not None:
                continue
            if r["tool"] not in best or r["quality"] > best[r["tool"]]["quality"]:
                best[r["tool"]] = {"user": r["user"], "helper": r["helper"], "quality": r["quality"]}
        result = {
            "knows_party": [i for i in ids if "party" in by_id[i].knowledge],
            "accepted_party": [i for i in ids if "party_accepted" in by_id[i].flags],
            "attended_party": [i for i in ids if by_id[i].counts.get("attend_party", 0)
                               or by_id[i].counts.get("host_party", 0)],
            "election_support": [i for i in ids if election.get(i) == "support"],
            "election_undecided": [i for i in ids if election.get(i) == "undecided"],
            "conversations": len(self.conversations),
            "tools": {"uses": len(self.tool_uses),
                      "helped": sum(r["helper"] is not None for r in self.tool_uses),
                      "learned": sum(len(r["learned"]) for r in self.tool_uses),
                      "best": {tool.id: best[tool.id] for tool in world.ALL_TOOLS if tool.id in best}},
            "build": self._build_summary(),
        }
        p = self.player
        if p is not None:
            told = [c for c in self.conversations
                    if c["topic"] == "party_word_of_mouth" and c["participants"][0] == PLAYER_ID]
            mine = [r["quality"] for r in self.tool_uses if PLAYER_ID in (r["user"], r["helper"])]
            built = [rid for rid in self.project_done if PLAYER_ID in (self.tool_uses[rid]["user"],
                                                                       self.tool_uses[rid]["helper"])]
            result["player"] = {
                "knows_party": "party" in p.knowledge,
                "accepted_party": "party_accepted" in p.flags,
                "attended_party": self.player_attended,
                "told_party": [c["participants"][1] for c in told],
                "told_party_accepted": [c["participants"][1] for c in told if c["effect"] == "accept"],
                "tool_uses": sum(r["user"] == PLAYER_ID for r in self.tool_uses),
                "helped": sum(r["helper"] == PLAYER_ID for r in self.tool_uses),
                "best_quality": max(mine) if mine else None,
                "ways_known": sum(len(s) for s in p.techniques.values()),
                "taught": sum(e["by"] == PLAYER_ID for r in self.tool_uses for e in r["learned"]),
                "built": len(built),
            }
        return result

    def _build_summary(self) -> dict:
        done = [self.tool_uses[rid] for rid in self.project_done]
        builders: list[str] = []
        for r in done:
            for agent_id in (r["user"], r["helper"]):
                if agent_id is not None and agent_id not in builders:
                    builders.append(agent_id)
        return {"stages_done": len(done), "finished": self.project_finished_t is not None,
                "finished_time": hhmm(self.project_finished_t) if self.project_finished_t is not None else None,
                "builders": builders,
                "quality": round(sum(r["quality"] for r in done) / len(done)) if done else None}


def build_replay(sim: Simulation, scorer: RewardScorer, model_name: str, device: str, wall_seconds: float) -> dict:
    stats = scorer.stats
    return {
        "meta": {
            "title": "Reward Town", "model": model_name, "device": device, "gpu": scorer.gpu_name,
            "dtype": str(scorer.dtype).removeprefix("torch."), "date": "2023-02-14",
            "weekday": "Tuesday", "tick_minutes": TICK_MINUTES, "num_ticks": sim.num_ticks,
            "start_view_tick": min(min(wake_tick(p) for p in world.PERSONAS), sim.num_ticks - 1),
            "rm_calls": stats["calls"], "rm_sequences": stats["sequences"],
            "rm_seconds": round(stats["seconds"], 3), "wall_seconds": round(wall_seconds, 3),
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": sim.summary(),
        },
        "map": {"width": world.MAP_W, "height": world.MAP_H,
                "locations": [location_dict(loc) for loc in world.LOCATIONS]},
        "agents": [{"id": p.id, "name": p.name, "initials": p.initials, "age": p.age, "color": p.color,
                    "home": p.home, "sprite": p.sprite, "description": p.description, "lifestyle": p.lifestyle}
                   for p in world.PERSONAS],
        "tools": [tool_dict(tool) for tool in world.ALL_TOOLS],
        "project": project_dict(),
        "ticks": sim.ticks,
        "decisions": sim.decisions,
        "conversations": sim.conversations,
        "tool_uses": sim.tool_uses,
    }


def location_dict(loc: world.Location) -> dict:
    return {"id": loc.id, "name": loc.name, "kind": loc.kind,
            "door": list(loc.door) if loc.door is not None else None,
            "area": list(loc.area) if loc.area is not None else None}
