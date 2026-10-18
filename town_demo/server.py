"""Reward Town live game server: serves town_demo/game/ and drives the NPCs with the reward model on demand.

Run from the repo root:  .venv-gpu/bin/python -m town_demo.server [--device cpu] [--port 8765]

One global game session per process (local demo); POST /api/new replaces it. All model calls happen under one
lock because a single model instance is shared across the HTTP threads.
"""

import argparse
import functools
import json
import re
import threading
import time
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import world
from .engine import (MAX_TICKS, TICK_MINUTES, PlayerTurn, Simulation, build_replay, location_dict, make_player,
                     project_dict, tool_dict)
from .scorer import RewardScorer

GAME_DIR = Path(__file__).resolve().parent / "game"
TOWN_JSON = GAME_DIR / "assets" / "town.json"
START_TICK = 42  # 07:00
MAX_BODY = 4096
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,11}$")
LOCK = threading.Lock()


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class GameSession:
    def __init__(self, scorer: RewardScorer, model_name: str, device: str):
        self.scorer = scorer
        self.model_name = model_name
        self.device = device
        self.phase = "intro"  # "intro" | "playing" | "ended"
        self.sim: Simulation | None = None
        self.avatar: str | None = None
        self.pending_gen = None
        self.pending_turn: PlayerTurn | None = None
        self.pending_kind: str | None = None  # "step" | "talk" | "tool"
        self.conv_cursor = 0
        self.tool_cursor = 0
        self.started = time.perf_counter()

    # ---- response pieces -------------------------------------------------------------------------------
    def meta(self) -> dict:
        s = self.scorer.stats
        return {"model": self.model_name, "device": self.device, "gpu": self.scorer.gpu_name,
                "dtype": str(self.scorer.dtype).removeprefix("torch."), "start_tick": START_TICK,
                "num_ticks": MAX_TICKS, "tick_minutes": TICK_MINUTES, "player_start": list(world.PLAYER_START),
                "rm_stats": {"calls": s["calls"], "sequences": s["sequences"], "seconds": round(s["seconds"], 3)}}

    def player_state(self) -> dict | None:
        if self.sim is None:
            return None
        p = self.sim.player
        return {"name": p.name, "avatar": self.avatar, "sprite": p.persona.sprite, "initials": p.persona.initials,
                "knowledge": sorted(p.knowledge), "flags": sorted(p.flags), "attended": self.sim.player_attended,
                "techniques": {tool_id: sorted(ids) for tool_id, ids in p.techniques.items()},
                "tool_ready": self.sim.player_tool_t != self.sim.t}

    def last_frame(self) -> dict | None:
        return self.sim.ticks[self.sim.t - 1] if self.sim is not None and self.sim.t > 0 else None

    def pending_view(self) -> dict | None:
        """The pending turn without judge scores or the model's pick: the player chooses blind."""
        turn = self.pending_turn
        if turn is None:
            return None
        return {"kind": turn.kind, "context": turn.context,
                "options": [{k: v for k, v in o.items() if k != "score"} for o in turn.options]}

    def new_conversations(self) -> list[dict]:
        convs = self.sim.conversations[self.conv_cursor:]
        self.conv_cursor = len(self.sim.conversations)
        return convs

    def new_tool_uses(self) -> list[dict]:
        ids = dict.fromkeys(self.sim.tool_events[self.tool_cursor:])
        self.tool_cursor = len(self.sim.tool_events)
        return [self.sim.tool_uses[i] for i in ids]

    def state(self) -> dict:
        sim = self.sim
        self.conv_cursor = len(sim.conversations) if sim is not None else 0
        self.tool_cursor = len(sim.tool_events) if sim is not None else 0
        return {
            "phase": self.phase, "meta": self.meta(),
            "locations": [location_dict(loc) for loc in world.LOCATIONS],
            "agents": [{"id": p.id, "name": p.name, "initials": p.initials, "age": p.age, "color": p.color,
                        "sprite": p.sprite, "description": p.description} for p in world.PERSONAS],
            "tools": [tool_dict(tool) for tool in world.ALL_TOOLS], "project": project_dict(),
            "player": self.player_state(), "frame": self.last_frame(), "pending": self.pending_view(),
            "conversations": sim.conversations if sim is not None else [],
            "tool_uses": sim.tool_uses if sim is not None else [],
            "summary": sim.summary() if sim is not None else None,
        }

    # ---- generator driving -----------------------------------------------------------------------------
    def start(self, gen, kind: str) -> dict:
        self.pending_gen, self.pending_kind = gen, kind
        return self.advance(None)

    def advance(self, value: int | None) -> dict:
        kind = self.pending_kind
        try:
            turn = self.pending_gen.send(value)
        except StopIteration:
            turn = None
        frame = None
        if turn is None:
            self.pending_gen = self.pending_turn = self.pending_kind = None
            if kind == "step":
                frame = self.last_frame()
                if self.sim.t >= self.sim.num_ticks:
                    self.phase = "ended"
        else:
            self.pending_turn = turn
        ended = self.phase == "ended"
        return {"pending": self.pending_view(), "frame": frame, "new_conversations": self.new_conversations(),
                "tool_uses": self.new_tool_uses(), "player": self.player_state(), "ended": ended,
                "summary": self.sim.summary() if ended else None}

    def require_playing(self) -> None:
        if self.phase != "playing":
            raise ApiError(409, "the day is over" if self.phase == "ended" else "no game in progress")
        if self.pending_gen is not None:
            raise ApiError(409, "a choice is waiting for you")

    def require_npc(self, npc_id) -> str:
        if self.sim is None:
            raise ApiError(409, "no game in progress")
        if not isinstance(npc_id, str) or npc_id not in world.PERSONA_BY_ID:
            raise ApiError(400, f"unknown npc {npc_id!r}")
        return npc_id

    # ---- endpoints -------------------------------------------------------------------------------------
    def new(self, body: dict) -> dict:
        name, avatar = body.get("name"), body.get("avatar")
        if not isinstance(name, str):
            raise ApiError(400, "name must be a string")
        name = " ".join(name.split())
        if not NAME_RE.match(name):
            raise ApiError(400, "name must start with a letter and use at most 12 letters, spaces, . ' -")
        if avatar not in world.PLAYER_SPRITES:
            raise ApiError(400, f"avatar must be one of {sorted(world.PLAYER_SPRITES)}")
        sim = Simulation(self.scorer, MAX_TICKS, make_player(name, avatar))
        sim.player_pos = world.PLAYER_START
        for _ in range(START_TICK):
            for _turn in sim.step():
                raise RuntimeError("player input requested during the pre-roll")
        self.sim, self.avatar, self.phase = sim, avatar, "playing"
        self.pending_gen = self.pending_turn = self.pending_kind = None
        self.started = time.perf_counter()
        return self.state()

    def step(self, body: dict) -> dict:
        self.require_playing()
        p = body.get("player")
        if not isinstance(p, dict):
            raise ApiError(400, "player must be an object {x, y, loc}")
        x, y, loc = p.get("x"), p.get("y"), p.get("loc")
        for v in (x, y):
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < world.MAP_W:
                raise ApiError(400, f"player x/y must be ints in [0, {world.MAP_W})")
        if loc is not None:
            if not isinstance(loc, str) or loc not in world.LOC_BY_ID:
                raise ApiError(400, f"unknown location {loc!r}")
            location = world.LOC_BY_ID[loc]
            if location.door is not None and (x, y) != location.door:
                raise ApiError(400, f"inside {loc} the player must be at its door {list(location.door)}")
            if location.area is not None:
                ax, ay, aw, ah = location.area
                if not (ax <= x < ax + aw and ay <= y < ay + ah):
                    raise ApiError(400, f"({x}, {y}) is outside {loc}")
        self.sim.player_pos = (x, y)
        self.sim.player.loc = loc
        return self.start(self.sim.step(), "step")

    def choose(self, body: dict) -> dict:
        if self.pending_turn is None:
            raise ApiError(409, "nothing is waiting for a choice")
        choice = body.get("choice")
        if not isinstance(choice, int) or isinstance(choice, bool) or not 0 <= choice < len(self.pending_turn.options):
            raise ApiError(400, f"choice must be an int in [0, {len(self.pending_turn.options)})")
        return self.advance(choice)

    def talk_options(self, npc_id) -> dict:
        return self.sim.player_topics(self.require_npc(npc_id))

    def talk(self, body: dict) -> dict:
        self.require_playing()
        npc_id = self.require_npc(body.get("npc"))
        topic_id = body.get("topic")
        options = self.sim.player_topics(npc_id)
        if not options["available"]:
            raise ApiError(400, options["reason"])
        topic = next((t for t in options["topics"] if t["id"] == topic_id), None)
        if topic is None:
            raise ApiError(400, f"unknown topic {topic_id!r}")
        if not topic["enabled"]:
            raise ApiError(400, topic["reason"])
        return self.start(self.sim.player_talk(npc_id, topic_id), "talk")

    def tools(self) -> dict:
        if self.sim is None:
            raise ApiError(409, "no game in progress")
        return self.sim.player_tools()

    def use_tool(self, body: dict) -> dict:
        self.require_playing()
        tool_id, helper_id = body.get("tool"), body.get("helper")
        if not isinstance(tool_id, str) or tool_id not in world.TOOL_BY_ID:
            raise ApiError(400, f"unknown tool {tool_id!r}")
        entry = next((t for t in self.sim.player_tools()["tools"] if t["id"] == tool_id), None)
        if entry is None:
            raise ApiError(400, f"The {world.TOOL_BY_ID[tool_id].name} is not here.")
        if not entry["can_use"]:
            raise ApiError(400, entry["use_reason"])
        if helper_id is not None:
            helper = next((h for h in entry["helpers"] if h["id"] == helper_id), None)
            if helper is None:
                raise ApiError(400, f"unknown helper {helper_id!r}" if helper_id not in world.PERSONA_BY_ID
                               else f"{world.PERSONA_BY_ID[helper_id].name} is not here.")
            if not helper["available"]:
                raise ApiError(400, helper["reason"])
        return self.start(self.sim.player_use_tool(tool_id, helper_id), "tool")

    def help(self, body: dict) -> dict:
        self.require_playing()
        use_id = body.get("use_id")
        if not isinstance(use_id, int) or isinstance(use_id, bool) or not 0 <= use_id < len(self.sim.tool_uses):
            raise ApiError(400, f"unknown tool use {use_id!r}")
        if use_id not in self.sim.active_uses.values():
            raise ApiError(400, "That job is already finished.")
        record = self.sim.tool_uses[use_id]
        if record["stage"] is not None:
            site = self.sim.player_site()
            if not site["can_help"]:
                raise ApiError(400, site["help_reason"])
            return self.start(self.sim.player_help(use_id), "tool")
        entry = next((t for t in self.sim.player_tools()["tools"] if t["id"] == record["tool"]), None)
        if entry is None:
            raise ApiError(400, f"The {world.TOOL_BY_ID[record['tool']].name} is not here.")
        if not entry["can_help"]:
            raise ApiError(400, entry["help_reason"])
        return self.start(self.sim.player_help(use_id), "tool")

    def site(self) -> dict:
        if self.sim is None:
            raise ApiError(409, "no game in progress")
        return self.sim.player_site()

    def build(self, body: dict) -> dict:
        self.require_playing()
        site = self.sim.player_site()
        if not site["can_build"]:
            raise ApiError(400, site["build_reason"])
        return self.start(self.sim.player_build(), "tool")

    def mind(self, npc_id) -> dict:
        npc_id = self.require_npc(npc_id)
        return next((d for d in reversed(self.sim.decisions) if d["agent"] == npc_id), {"none": True})

    def log(self) -> dict:
        if self.sim is None:
            raise ApiError(409, "no game in progress")
        return build_replay(self.sim, self.scorer, self.model_name, self.device, time.perf_counter() - self.started)


SESSION: GameSession | None = None


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self.api("GET")
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/"):
            self.api("POST")
        else:
            self.send_json(405, {"error": "method not allowed"})

    def log_request(self, code="-", size="-") -> None:
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(400, "bad Content-Length") from None
        if length > MAX_BODY:
            raise ApiError(413, f"request body over {MAX_BODY} bytes")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ApiError(400, f"bad JSON: {err}") from None
        if not isinstance(body, dict):
            raise ApiError(400, "JSON body must be an object")
        return body

    def api(self, method: str) -> None:
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        s = SESSION
        routes = {
            ("GET", "/api/state"): lambda: s.state(),
            ("POST", "/api/new"): lambda: s.new(self.read_body()),
            ("POST", "/api/step"): lambda: s.step(self.read_body()),
            ("POST", "/api/choose"): lambda: s.choose(self.read_body()),
            ("GET", "/api/talk-options"): lambda: s.talk_options(query.get("npc")),
            ("POST", "/api/talk"): lambda: s.talk(self.read_body()),
            ("GET", "/api/tools"): lambda: s.tools(),
            ("POST", "/api/use-tool"): lambda: s.use_tool(self.read_body()),
            ("POST", "/api/help"): lambda: s.help(self.read_body()),
            ("GET", "/api/site"): lambda: s.site(),
            ("POST", "/api/build"): lambda: s.build(self.read_body()),
            ("GET", "/api/mind"): lambda: s.mind(query.get("npc")),
            ("GET", "/api/log"): lambda: s.log(),
        }
        handler = routes.get((method, url.path))
        if handler is None:
            self.send_json(404, {"error": f"unknown API {method} {url.path}"})
            return
        try:
            with LOCK:
                payload = handler()
        except ApiError as err:
            self.send_json(err.status, {"error": err.message})
            return
        except Exception as err:  # keep the server alive; the client shows the message
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(err).__name__}: {err}"})
            return
        self.send_json(HTTPStatus.OK, payload)


def main() -> None:
    global SESSION
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models/Skywork-Reward-V2-Qwen3-0.6B")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    world.validate()
    if not TOWN_JSON.is_file():
        raise SystemExit("Missing game assets; run: .venv-gpu/bin/python -m town_demo.fetch_assets")
    scorer = RewardScorer(args.model, args.device, args.batch_size)
    SESSION = GameSession(scorer, args.model.name, args.device)
    server = ThreadingHTTPServer((args.host, args.port), functools.partial(Handler, directory=str(GAME_DIR)))
    print(f"Reward Town: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
