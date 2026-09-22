from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActionType(StrEnum):
    WORK = "work"
    REST = "rest"
    BUY_FOOD = "buy_food"
    GIVE = "give"
    STEAL = "steal"
    TALK = "talk"


@dataclass
class AgentProfile:
    """Scenario-agnostic psychological starting point for an agent."""

    traits: dict[str, float] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)


@dataclass
class Memory:
    id: str
    content: str
    subject: str | None
    source_event_kind: str
    importance: float
    emotional_intensity: float
    confidence: float
    created_turn: int
    last_recalled: int
    recall_count: int = 0


@dataclass
class Belief:
    subject: str
    statement: str
    confidence: float
    updated_turn: int


@dataclass
class Agent:
    name: str
    money: int = 100
    hunger: int = 10
    energy: int = 100
    food: int = 1
    reputation: int = 50
    profile: AgentProfile = field(default_factory=AgentProfile)
    working_memory: list[str] = field(default_factory=list)
    memories: list[Memory] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    relationships: dict[str, float] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.hunger < 100


@dataclass(frozen=True)
class Action:
    actor: str
    kind: ActionType
    target: str | None = None
    amount: int = 0
    message: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Event:
    turn: int
    actor: str
    kind: str
    summary: str
    target: str | None = None
    amount: int = 0
    reason: str | None = None
    message: str | None = None
