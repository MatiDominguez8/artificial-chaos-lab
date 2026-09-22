from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import Agent, Belief, Event, Memory


@dataclass(frozen=True)
class MemoryDecision:
    remember: bool
    content: str = ""
    subject: str | None = None
    importance: float = 0.0
    emotional_intensity: float = 0.0
    confidence: float = 1.0
    relationship_delta: float = 0.0
    belief: str | None = None
    belief_confidence: float = 0.0


class MemoryInterpreter(Protocol):
    def interpret(self, observer: Agent, event: Event) -> MemoryDecision: ...


class HeuristicMemoryInterpreter:
    """Deterministic baseline until an LLM interprets events subjectively.

    It intentionally keeps the mechanism separate from the rules of the world.
    Later an LLM can replace this class without changing the event log or the
    simulation engine.
    """

    def interpret(self, observer: Agent, event: Event) -> MemoryDecision:
        is_actor = observer.name == event.actor
        is_target = observer.name == event.target
        counterpart = event.target if is_actor else event.actor if is_target else None

        if not (is_actor or is_target):
            return MemoryDecision(remember=False)

        if event.kind == "steal":
            if is_target:
                return MemoryDecision(
                    remember=True,
                    content=f"{event.actor} stole {event.amount} coins from me.",
                    subject=event.actor,
                    importance=0.95,
                    emotional_intensity=0.95,
                    confidence=1.0,
                    relationship_delta=-0.35,
                    belief=f"{event.actor} may exploit me when given the chance.",
                    belief_confidence=0.85,
                )
            return MemoryDecision(
                remember=True,
                content=f"I stole {event.amount} coins from {event.target}.",
                subject=event.target,
                importance=0.80,
                emotional_intensity=0.70,
                confidence=1.0,
            )

        if event.kind == "failed_steal":
            if is_target:
                return MemoryDecision(
                    remember=True,
                    content=f"{event.actor} tried to steal from me and failed.",
                    subject=event.actor,
                    importance=0.88,
                    emotional_intensity=0.85,
                    relationship_delta=-0.25,
                    belief=f"{event.actor} is willing to steal from me.",
                    belief_confidence=0.90,
                )
            return MemoryDecision(
                remember=True,
                content=f"I tried to steal from {event.target} and failed.",
                subject=event.target,
                importance=0.65,
                emotional_intensity=0.60,
            )

        if event.kind == "give":
            if is_target:
                return MemoryDecision(
                    remember=True,
                    content=f"{event.actor} gave me {event.amount} coins.",
                    subject=event.actor,
                    importance=0.72,
                    emotional_intensity=0.55,
                    relationship_delta=0.20,
                    belief=f"{event.actor} has helped me before.",
                    belief_confidence=0.75,
                )
            return MemoryDecision(
                remember=True,
                content=f"I gave {event.amount} coins to {event.target}.",
                subject=event.target,
                importance=0.50,
                emotional_intensity=0.30,
            )

        if event.kind == "talk" and counterpart:
            return MemoryDecision(
                remember=False,
                subject=counterpart,
                relationship_delta=0.02,
            )

        if event.kind == "collapse" and is_actor:
            return MemoryDecision(
                remember=True,
                content="I collapsed from hunger.",
                subject=None,
                importance=0.95,
                emotional_intensity=0.90,
            )

        return MemoryDecision(remember=False)


class CognitionEngine:
    """Turns objective events into each agent's subjective internal state."""

    def __init__(
        self,
        interpreter: MemoryInterpreter | None = None,
        working_memory_limit: int = 8,
    ) -> None:
        self.interpreter = interpreter or HeuristicMemoryInterpreter()
        self.working_memory_limit = working_memory_limit

    def observe(self, event: Event, agents: dict[str, Agent]) -> None:
        observer_names = {event.actor}
        if event.target:
            observer_names.add(event.target)

        for name in observer_names:
            observer = agents.get(name)
            if observer is None:
                continue
            self._remember_recent_observation(observer, event)
            decision = self.interpreter.interpret(observer, event)
            self._apply_decision(observer, event, decision)

    def _remember_recent_observation(
        self,
        observer: Agent,
        event: Event,
    ) -> None:
        observer.working_memory.append(
            f"Turn {event.turn}: {event.summary}"
        )
        if len(observer.working_memory) > self.working_memory_limit:
            del observer.working_memory[:-self.working_memory_limit]

    def retrieve(
        self,
        agent: Agent,
        current_turn: int,
        subject: str | None = None,
        limit: int = 5,
    ) -> list[Memory]:
        if limit <= 0:
            return []

        ranked = sorted(
            agent.memories,
            key=lambda memory: self._score(memory, current_turn, subject),
            reverse=True,
        )[:limit]

        for memory in ranked:
            memory.last_recalled = current_turn
            memory.recall_count += 1

        return ranked

    def _apply_decision(
        self,
        observer: Agent,
        event: Event,
        decision: MemoryDecision,
    ) -> None:
        if decision.subject and decision.relationship_delta:
            current = observer.relationships.get(decision.subject, 0.0)
            observer.relationships[decision.subject] = self._clamp(
                current + decision.relationship_delta,
                -1.0,
                1.0,
            )

        if decision.belief and decision.subject:
            self._upsert_belief(
                observer,
                subject=decision.subject,
                statement=decision.belief,
                confidence=decision.belief_confidence,
                turn=event.turn,
            )

        if not decision.remember:
            return

        memory_id = (
            f"{event.turn}:{observer.name}:{event.kind}:"
            f"{event.actor}:{event.target or '-'}:{len(observer.memories)}"
        )
        observer.memories.append(
            Memory(
                id=memory_id,
                content=decision.content,
                subject=decision.subject,
                source_event_kind=event.kind,
                importance=self._clamp(decision.importance, 0.0, 1.0),
                emotional_intensity=self._clamp(
                    decision.emotional_intensity,
                    0.0,
                    1.0,
                ),
                confidence=self._clamp(decision.confidence, 0.0, 1.0),
                created_turn=event.turn,
                last_recalled=event.turn,
            )
        )

    def _upsert_belief(
        self,
        agent: Agent,
        subject: str,
        statement: str,
        confidence: float,
        turn: int,
    ) -> None:
        for belief in agent.beliefs:
            if belief.subject == subject and belief.statement == statement:
                belief.confidence = self._clamp(
                    max(belief.confidence, confidence),
                    0.0,
                    1.0,
                )
                belief.updated_turn = turn
                return

        agent.beliefs.append(
            Belief(
                subject=subject,
                statement=statement,
                confidence=self._clamp(confidence, 0.0, 1.0),
                updated_turn=turn,
            )
        )

    @staticmethod
    def _score(memory: Memory, current_turn: int, subject: str | None) -> float:
        age = max(0, current_turn - memory.created_turn)
        recency = 1.0 / (1.0 + age / 25.0)
        recalled = min(memory.recall_count / 5.0, 1.0)
        subject_match = 1.0 if subject and memory.subject == subject else 0.0

        return (
            0.40 * memory.importance
            + 0.25 * memory.emotional_intensity
            + 0.15 * recency
            + 0.10 * recalled
            + 0.10 * subject_match
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
