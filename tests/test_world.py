import random

from src.acl.cognition import CognitionEngine
from src.acl.models import Action, ActionType, Agent, Memory
from src.acl.world import World


class WorkPolicy:
    def choose_action(self, agent, others):
        return Action(agent.name, ActionType.WORK)


class OneWayStealPolicy:
    def choose_action(self, agent, others):
        if agent.name == "Ada":
            return Action("Ada", ActionType.STEAL, target="Bruno", amount=10)
        return Action(agent.name, ActionType.REST)


def test_work_changes_money_and_turn():
    rng = random.Random(1)
    world = World([Agent("Ada"), Agent("Bruno")], WorkPolicy(), rng)
    world.step()

    assert world.turn == 1
    assert world.agents["Ada"].money == 112
    assert world.agents["Bruno"].money == 112
    assert any(event.kind == "work" for event in world.events)


def test_same_objective_event_creates_different_subjective_memories():
    rng = random.Random(1)  # First steal succeeds (< 0.35).
    world = World([Agent("Ada"), Agent("Bruno")], OneWayStealPolicy(), rng)
    world.step()

    ada = world.agents["Ada"]
    bruno = world.agents["Bruno"]

    assert any(memory.content == "I stole 10 coins from Bruno." for memory in ada.memories)
    assert any(memory.content == "Ada stole 10 coins from me." for memory in bruno.memories)
    assert bruno.relationships["Ada"] == -0.35
    assert any(
        belief.subject == "Ada" and "exploit" in belief.statement
        for belief in bruno.beliefs
    )


def test_recall_prefers_relevant_emotional_memory_and_reinforces_it():
    agent = Agent(
        "Pedro",
        memories=[
            Memory(
                id="1",
                content="Bruno refused to help me.",
                subject="Bruno",
                source_event_kind="refusal",
                importance=0.9,
                emotional_intensity=0.9,
                confidence=1.0,
                created_turn=10,
                last_recalled=10,
            ),
            Memory(
                id="2",
                content="I bought bread.",
                subject=None,
                source_event_kind="buy_food",
                importance=0.2,
                emotional_intensity=0.1,
                confidence=1.0,
                created_turn=19,
                last_recalled=19,
            ),
        ],
    )

    cognition = CognitionEngine()
    recalled = cognition.retrieve(agent, current_turn=20, subject="Bruno", limit=1)

    assert recalled[0].id == "1"
    assert recalled[0].recall_count == 1
    assert recalled[0].last_recalled == 20


def test_seeded_random_runs_are_reproducible():
    from src.acl.simulation import run

    first = run(turns=20, seed=42)
    second = run(turns=20, seed=42)

    assert first.events == second.events
    assert first.agents == second.agents
