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
class Agent:
    name: str
    money: int = 100
    hunger: int = 10
    energy: int = 100
    food: int = 1
    reputation: int = 50
    memories: list[str] = field(default_factory=list)

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


@dataclass(frozen=True)
class Event:
    turn: int
    actor: str
    kind: str
    summary: str
    target: str | None = None
    amount: int = 0
