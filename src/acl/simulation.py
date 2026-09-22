from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .models import Agent, AgentProfile
from .policies import Policy, RandomPolicy
from .world import World


def tiny_town_agents() -> list[Agent]:
    """Deliberately varied cast used to test whether profiles affect behavior.

    These personalities belong to Experiment 001 only. They are not assumptions
    baked into the simulation engine.
    """

    profiles = {
        "Ada": AgentProfile(
            traits={
                "risk_tolerance": 0.15,
                "sociability": 0.35,
                "generosity": 0.55,
                "impulsivity": 0.10,
                "competitiveness": 0.20,
                "trustfulness": 0.55,
                "curiosity": 0.30,
                "rule_respect": 0.90,
            },
            goals=[
                "Maintain personal security.",
                "Avoid unnecessary conflict.",
            ],
        ),
        "Bruno": AgentProfile(
            traits={
                "risk_tolerance": 0.55,
                "sociability": 0.40,
                "generosity": 0.15,
                "impulsivity": 0.35,
                "competitiveness": 0.90,
                "trustfulness": 0.25,
                "curiosity": 0.45,
                "rule_respect": 0.55,
            },
            goals=[
                "Accumulate resources.",
                "Do better than the others.",
            ],
        ),
        "Cora": AgentProfile(
            traits={
                "risk_tolerance": 0.30,
                "sociability": 0.95,
                "generosity": 0.90,
                "impulsivity": 0.25,
                "competitiveness": 0.15,
                "trustfulness": 0.80,
                "curiosity": 0.65,
                "rule_respect": 0.80,
            },
            goals=[
                "Build strong relationships.",
                "Help people who seem to need it.",
            ],
        ),
        "Dante": AgentProfile(
            traits={
                "risk_tolerance": 0.95,
                "sociability": 0.70,
                "generosity": 0.35,
                "impulsivity": 0.90,
                "competitiveness": 0.55,
                "trustfulness": 0.45,
                "curiosity": 0.95,
                "rule_respect": 0.20,
            },
            goals=[
                "Seek novelty and excitement.",
                "Avoid boring repetitive behavior.",
            ],
        ),
        "Eva": AgentProfile(
            traits={
                "risk_tolerance": 0.60,
                "sociability": 0.85,
                "generosity": 0.40,
                "impulsivity": 0.45,
                "competitiveness": 0.75,
                "trustfulness": 0.50,
                "curiosity": 0.70,
                "rule_respect": 0.60,
            },
            goals=[
                "Gain influence.",
                "Build useful alliances.",
            ],
        ),
    }

    return [Agent(name=name, profile=profile) for name, profile in profiles.items()]


def build_world(seed: int = 7, policy: Policy | None = None) -> World:
    rng = random.Random(seed)
    return World(tiny_town_agents(), policy or RandomPolicy(rng), rng)


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
