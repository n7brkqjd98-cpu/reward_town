"""Headless Reward Town day (engine regression/benchmark): every choice is the reward model's argmax.

Run from the repo root:  .venv-gpu/bin/python -m town_demo.simulate
"""

import argparse
import json
import os
import time
from pathlib import Path

from . import world
from .engine import MAX_TICKS, Simulation, build_replay
from .scorer import RewardScorer


def tick_count(value: str) -> int:
    n = int(value)
    if not 1 <= n <= MAX_TICKS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_TICKS}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models/Skywork-Reward-V2-Qwen3-0.6B")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ticks", type=tick_count, default=MAX_TICKS)
    parser.add_argument("--out", type=Path, default=Path("out/town-demo"))
    args = parser.parse_args()

    world.validate()
    start = time.perf_counter()
    scorer = RewardScorer(args.model, args.device, args.batch_size)
    sim = Simulation(scorer, args.ticks)
    sim.run()
    wall = time.perf_counter() - start

    replay = build_replay(sim, scorer, args.model.name, args.device, wall)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "replay.json"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False)
    os.replace(tmp, path)

    print(json.dumps(replay["meta"]["summary"], ensure_ascii=False))
    print(f"wrote {path}")
    print(f"wall time {wall:.1f}s")


if __name__ == "__main__":
    main()
