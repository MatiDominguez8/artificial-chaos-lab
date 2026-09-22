from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .models import Agent
from .policies import Policy, RandomPolicy
from .world import World


def build_world(seed: int = 7, policy: Policy | None = None) -> World:
    rng = random.Random(seed)
    names = ["Ada", "Bruno", "Cora", "Dante", "Eva"]
    agents = [Agent(name=name) for name in names]
    return World(agents, policy or RandomPolicy(rng), rng)


def run(
    turns: int = 100,
    seed: int = 7,
    log_path: Path | None = None,
    policy: Policy | None = None,
) -> World:
    world = build_world(seed, policy=policy)
    handle = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        for _ in range(turns):
            for event in world.step():
                if handle:
                    handle.write(
                        json.dumps(asdict(event), ensure_ascii=False) + "\n"
                    )
    finally:
        if handle:
            handle.close()
    return world
