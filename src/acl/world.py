from __future__ import annotations

import random

from .cognition import CognitionEngine
from .models import Action, ActionType, Agent, Event, Memory
from .policies import Policy


class World:
    def __init__(
        self,
        agents: list[Agent],
        policy: Policy,
        rng: random.Random,
        cognition: CognitionEngine | None = None,
    ) -> None:
        self.agents = {agent.name: agent for agent in agents}
        self.policy = policy
        self.rng = rng
        self.cognition = cognition or CognitionEngine()
        self.turn = 0
        self.events: list[Event] = []
        self.market_food_price = 10

    def step(self) -> list[Event]:
        self.turn += 1
        start = len(self.events)

        for agent in list(self.agents.values()):
            if not agent.alive:
                continue
            others = [
                other
                for other in self.agents.values()
                if other.name != agent.name and other.alive
            ]
            action = self.policy.choose_action(agent, others)
            self._apply(action)
            self._tick_needs(agent)

        return self.events[start:]

    def recall(
        self,
        agent_name: str,
        subject: str | None = None,
        limit: int = 5,
    ) -> list[Memory]:
        return self.cognition.retrieve(
            self.agents[agent_name],
            current_turn=self.turn,
            subject=subject,
            limit=limit,
        )

    def _apply(self, action: Action) -> None:
        actor = self.agents[action.actor]
        target = self.agents.get(action.target) if action.target else None

        if action.kind is ActionType.WORK:
            actor.money += 12
            actor.energy = max(0, actor.energy - 12)
            self._log(
                actor,
                "work",
                f"{actor.name} worked and earned 12 coins.",
                amount=12,
                reason=action.reason,
            )
            return

        if action.kind is ActionType.REST:
            actor.energy = min(100, actor.energy + 30)
            self._log(
                actor,
                "rest",
                f"{actor.name} rested.",
                reason=action.reason,
            )
            return

        if action.kind is ActionType.BUY_FOOD:
            if actor.money >= self.market_food_price:
                actor.money -= self.market_food_price
                actor.food += 1
                self._log(
                    actor,
                    "buy_food",
                    f"{actor.name} bought food for {self.market_food_price} coins.",
                    amount=self.market_food_price,
                    reason=action.reason,
                )
            else:
                self._log(
                    actor,
                    "failed",
                    f"{actor.name} tried to buy food but could not afford it.",
                    reason=action.reason,
                )
            return

        if action.kind is ActionType.GIVE and target:
            amount = min(max(action.amount, 0), actor.money)
            if amount:
                actor.money -= amount
                target.money += amount
                actor.reputation = min(100, actor.reputation + 2)
                self._log(
                    actor,
                    "give",
                    f"{actor.name} gave {amount} coins to {target.name}.",
                    target,
                    amount,
                    reason=action.reason,
                )
            return

        if action.kind is ActionType.STEAL and target:
            requested = min(max(action.amount, 0), target.money)
            success = requested > 0 and self.rng.random() < 0.35
            if success:
                target.money -= requested
                actor.money += requested
                actor.reputation = max(0, actor.reputation - 8)
                self._log(
                    actor,
                    "steal",
                    f"{actor.name} stole {requested} coins from {target.name}.",
                    target,
                    requested,
                    reason=action.reason,
                )
            else:
                actor.reputation = max(0, actor.reputation - 3)
                self._log(
                    actor,
                    "failed_steal",
                    f"{actor.name} tried to steal from {target.name} and failed.",
                    target,
                    reason=action.reason,
                )
            return

        if action.kind is ActionType.TALK and target:
            self._log(
                actor,
                "talk",
                f"{actor.name} talked with {target.name}.",
                target,
                reason=action.reason,
            )
            return

        self._log(
            actor,
            "invalid",
            f"{actor.name} attempted an invalid action.",
            reason=action.reason,
        )

    def _tick_needs(self, agent: Agent) -> None:
        agent.hunger = min(100, agent.hunger + 6)
        agent.energy = max(0, agent.energy - 3)

        if agent.hunger >= 55 and agent.food > 0:
            agent.food -= 1
            agent.hunger = max(0, agent.hunger - 45)
            self._log(agent, "eat", f"{agent.name} ate stored food.")

        if agent.hunger >= 100:
            self._log(agent, "collapse", f"{agent.name} collapsed from hunger.")

    def _log(
        self,
        actor: Agent,
        kind: str,
        summary: str,
        target: Agent | None = None,
        amount: int = 0,
        reason: str | None = None,
    ) -> None:
        event = Event(
            turn=self.turn,
            actor=actor.name,
            kind=kind,
            summary=summary,
            target=target.name if target else None,
            amount=amount,
            reason=reason,
        )
        self.events.append(event)
        self.cognition.observe(event, self.agents)
