from __future__ import annotations

import random
from typing import Any, Protocol

from .models import Action, ActionType, Agent
from .ollama_client import OllamaClient


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
            return Action(
                agent.name,
                kind,
                target=target,
                message="Small talk with unknown consequences.",
            )
        return Action(agent.name, kind)


class OllamaPolicy:
    """Local LLM policy.

    The model chooses an intention. The World remains authoritative and applies
    all state changes, probabilities and consequences.
    """

    TARGET_ACTIONS = {
        ActionType.TALK,
        ActionType.GIVE,
        ActionType.STEAL,
    }

    def __init__(
        self,
        *,
        model: str = "qwen3:14b",
        client: OllamaClient | None = None,
        temperature: float = 0.8,
        num_ctx: int = 4096,
        memory_limit: int = 5,
    ) -> None:
        self.model = model
        self.client = client or OllamaClient()
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.memory_limit = memory_limit

    def choose_action(self, agent: Agent, others: list[Agent]) -> Action:
        schema = self._schema(others)
        result = self.client.chat(
            model=self.model,
            messages=self._messages(agent, others),
            schema=schema,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )
        return self._to_action(agent, others, result)

    def _messages(
        self,
        agent: Agent,
        others: list[Agent],
    ) -> list[dict[str, str]]:
        people = []
        for other in others:
            relationship = agent.relationships.get(other.name, 0.0)
            people.append(f"- {other.name}: relationship={relationship:+.2f}")

        memories = sorted(
            agent.memories,
            key=lambda item: (
                item.importance,
                item.emotional_intensity,
                item.recall_count,
                item.created_turn,
            ),
            reverse=True,
        )[: self.memory_limit]

        memory_lines = [
            (
                f"- {memory.content} "
                f"(importance={memory.importance:.2f}, "
                f"emotion={memory.emotional_intensity:.2f}, "
                f"confidence={memory.confidence:.2f})"
            )
            for memory in memories
        ]

        belief_lines = [
            f"- About {belief.subject}: {belief.statement} "
            f"(confidence={belief.confidence:.2f})"
            for belief in agent.beliefs[-5:]
        ]

        trait_lines = [
            f"- {name}: {value:.2f}"
            for name, value in sorted(agent.profile.traits.items())
        ]
        goal_lines = [f"- {goal}" for goal in agent.profile.goals]
        recent_lines = [f"- {item}" for item in agent.working_memory]

        system = """You control one person inside a simulation.
You are NOT the simulation engine and you are NOT omniscient.
Choose one action using only the information you are given.
Do not invent resources, people, rules, actions or hidden facts.
Your decision may be selfish, cooperative, impulsive or cautious; do not try to make the society succeed.
Personality traits are tendencies, not hard rules. Goals can compete with each other.
Do not act like an optimal game-playing bot unless the supplied personality actually points that way.

Known mechanics:
- work: requires at least 15 energy; earn 12 coins and spend 12 energy.
- rest: recover 30 energy.
- buy_food: costs 10 coins and adds 1 stored food.
- give: transfer some of your coins to another person and slightly improves reputation.
- steal: attempt to take coins from another person; it can fail and harms reputation.
- talk: speak to another person; the exact message matters because the other person interprets it subjectively.
- every turn also increases hunger by 6 and reduces energy by 3.
- when hunger reaches 55 and stored food exists, one food is automatically eaten and hunger drops by 45.
- hunger at 100 means collapse.

For talk, give and steal, target must be one of the listed people.
For work, rest and buy_food, target must be an empty string.
Return only the structured JSON requested by the schema.
Keep reason short. Do not write chain-of-thought."""

        user = f"""You are {agent.name}.

YOUR PRIVATE STATE
money: {agent.money}
hunger: {agent.hunger}/100
energy: {agent.energy}/100
stored food: {agent.food}
reputation: {agent.reputation}/100

YOUR STARTING PERSONALITY
{chr(10).join(trait_lines) if trait_lines else "- no strong predefined traits"}

YOUR CURRENT GOALS
{chr(10).join(goal_lines) if goal_lines else "- no explicit goals"}

WHAT YOU RECENTLY EXPERIENCED
{chr(10).join(recent_lines) if recent_lines else "- nothing yet"}

PEOPLE YOU CURRENTLY KNOW
{chr(10).join(people) if people else "- nobody"}

YOUR RELEVANT MEMORIES
{chr(10).join(memory_lines) if memory_lines else "- none yet"}

YOUR CURRENT BELIEFS
{chr(10).join(belief_lines) if belief_lines else "- none yet"}

Choose exactly one action."""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _schema(self, others: list[Agent]) -> dict[str, Any]:
        names = [""] + [agent.name for agent in others]
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [kind.value for kind in ActionType],
                },
                "target": {
                    "type": "string",
                    "enum": names,
                },
                "amount": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                },
                "message": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["action", "target", "amount", "message", "reason"],
        }

    def _to_action(
        self,
        agent: Agent,
        others: list[Agent],
        result: dict[str, Any],
    ) -> Action:
        try:
            kind = ActionType(str(result["action"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Model returned invalid action: {result!r}") from exc

        target = str(result.get("target", "")).strip() or None
        valid_targets = {other.name for other in others}

        if kind in self.TARGET_ACTIONS:
            if target not in valid_targets:
                raise ValueError(
                    f"{kind.value} requires a valid target; got {target!r}"
                )
        else:
            target = None

        try:
            amount = int(result.get("amount", 0))
        except (TypeError, ValueError):
            amount = 0

        amount = max(0, min(amount, 20))
        if kind is ActionType.GIVE and amount == 0:
            amount = 5
        if kind is ActionType.STEAL and amount == 0:
            amount = 10

        message = str(result.get("message", "")).strip() or None
        reason = str(result.get("reason", "")).strip() or None

        return Action(
            actor=agent.name,
            kind=kind,
            target=target,
            amount=amount,
            message=message,
            reason=reason,
        )
