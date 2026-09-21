import random

from src.acl.models import Action, ActionType, Agent
from src.acl.world import World


class WorkPolicy:
    def choose_action(self, agent, others):
        return Action(agent.name, ActionType.WORK)


def test_work_changes_money_and_turn():
    rng = random.Random(1)
    world = World([Agent("Ada"), Agent("Bruno")], WorkPolicy(), rng)
    world.step()

    assert world.turn == 1
    assert world.agents["Ada"].money == 112
    assert world.agents["Bruno"].money == 112
    assert any(event.kind == "work" for event in world.events)


def test_seeded_random_runs_are_reproducible():
    from src.acl.simulation import run

    first = run(turns=20, seed=42)
    second = run(turns=20, seed=42)

    assert first.events == second.events
    assert first.agents == second.agents
