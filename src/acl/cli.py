from __future__ import annotations

import argparse
from pathlib import Path

from .cognition import CognitionEngine, OllamaMemoryInterpreter
from .ollama_client import OllamaClient
from .policies import OllamaPolicy
from .simulation import run


def build_console_summary(
    world,
    *,
    event_log: Path,
    cognition_log: Path,
    console_log: Path,
) -> str:
    lines = [f"Finished {world.turn} turns. Events: {len(world.events)}"]
    for agent in world.agents.values():
        lines.append(
            f"{agent.name:>6} | money={agent.money:>4} | "
            f"hunger={agent.hunger:>3} | energy={agent.energy:>3} "
            f"| rep={agent.reputation:>3} | alive={agent.alive}"
        )
    lines.append(f"Log: {event_log}")
    lines.append(f"Cognition log: {cognition_log}")
    lines.append(f"Console log: {console_log}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Artificial Chaos Lab experiment"
    )
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log", type=Path, default=Path("logs/latest.jsonl"))
    parser.add_argument(
        "--cognition-log",
        type=Path,
        default=Path("logs/latest_cognition.jsonl"),
    )
    parser.add_argument(
        "--console-log",
        type=Path,
        default=Path("logs/latest_console.log"),
    )
    parser.add_argument(
        "--policy",
        choices=("random", "ollama"),
        default="random",
        help="Decision policy used by all agents.",
    )
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()

    policy = None
    cognition = None
    if args.policy == "ollama":
        client = OllamaClient(base_url=args.ollama_url)
        policy = OllamaPolicy(
            model=args.model,
            client=client,
            temperature=args.temperature,
            num_ctx=args.context,
        )
        cognition = CognitionEngine(
            interpreter=OllamaMemoryInterpreter(
                client=client,
                model=args.model,
                temperature=0.7,
                num_ctx=args.context,
            )
        )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.cognition_log.parent.mkdir(parents=True, exist_ok=True)
    args.console_log.parent.mkdir(parents=True, exist_ok=True)

    world = run(
        turns=args.turns,
        seed=args.seed,
        log_path=args.log,
        policy=policy,
        cognition=cognition,
        cognition_log_path=args.cognition_log,
    )

    summary = build_console_summary(
        world,
        event_log=args.log,
        cognition_log=args.cognition_log,
        console_log=args.console_log,
    )
    print(summary)
    args.console_log.write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
