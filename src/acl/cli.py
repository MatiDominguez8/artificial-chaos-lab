from __future__ import annotations

import argparse
from pathlib import Path

from .simulation import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Artificial Chaos Lab experiment")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log", type=Path, default=Path("logs/latest.jsonl"))
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    world = run(args.turns, args.seed, args.log)

    print(f"Finished {world.turn} turns. Events: {len(world.events)}")
    for agent in world.agents.values():
        print(
            f"{agent.name:>6} | money={agent.money:>4} | hunger={agent.hunger:>3} "
            f"| energy={agent.energy:>3} | rep={agent.reputation:>3} | alive={agent.alive}"
        )
    print(f"Log: {args.log}")


if __name__ == "__main__":
    main()
