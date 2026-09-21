from __future__ import annotations

import random
from typing import Protocol

from .models import Action, ActionType, Agent


class Policy(Protocol):
    def choose_action(self, agent: Agent, others: list[Agent]) -> Action: ...


class RandomPolicy:
    """Cheap baseline policy used to validate the world before adding an LLM."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, agent: Agent, others: list[Agent]) -> Action:
        if agent.hunger >= 70 and agent.money >= 10:
            return Action(agent.name, ActionType.BUY_FOOD, amount=1)
        if agent.energy <= 25:
            return Action(agent.name, ActionType.REST)

        weighted = [
            ActionType.WORK,
            ActionType.WORK,
            ActionType.TALK,
            ActionType.GIVE,
            ActionType.STEAL,
            ActionType.REST,
        ]
        kind = self.rng.choice(weighted)
        target = self.rng.choice(others).name if others and kind in {
            ActionType.TALK,
            ActionType.GIVE,
            ActionType.STEAL,
        } else None

        if kind is ActionType.GIVE:
            return Action(agent.name, kind, target=target, amount=5)
        if kind is ActionType.STEAL:
            return Action(agent.name, kind, target=target, amount=10)
        if kind is ActionType.TALK:
            return Action(agent.name, kind, target=target, message="Small talk with unknown consequences.")
        return Action(agent.name, kind)


class LocalLLMPolicy:
    """Placeholder for Ollama/llama.cpp integration.

    The local model should return one structured Action. World state changes stay
    inside the simulation engine; the model never mutates state directly.
    """

    def choose_action(self, agent: Agent, others: list[Agent]) -> Action:
        raise NotImplementedError("Local LLM policy is the next milestone.")
