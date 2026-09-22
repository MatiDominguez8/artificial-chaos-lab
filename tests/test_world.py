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



def test_tiny_town_agents_have_distinct_scenario_agnostic_profiles():
    from src.acl.simulation import tiny_town_agents

    agents = tiny_town_agents()
    profiles = [agent.profile for agent in agents]

    assert len({tuple(sorted(profile.traits.items())) for profile in profiles}) == 5
    assert all(profile.goals for profile in profiles)

    forbidden_scenario_traits = {"money", "hunger", "energy", "food", "reputation"}
    for profile in profiles:
        assert forbidden_scenario_traits.isdisjoint(profile.traits)



def test_working_memory_tracks_recent_direct_experience_and_is_bounded():
    rng = random.Random(1)
    world = World(
        [Agent("Ada"), Agent("Bruno")],
        WorkPolicy(),
        rng,
        cognition=CognitionEngine(working_memory_limit=2),
    )

    world.step()
    world.step()
    world.step()

    ada = world.agents["Ada"]
    assert len(ada.working_memory) == 2
    assert ada.working_memory[0].startswith("Turn 2:")
    assert ada.working_memory[1].startswith("Turn 3:")



class TalkPolicy:
    def choose_action(self, agent, others):
        if agent.name == "Ada":
            return Action(
                "Ada",
                ActionType.TALK,
                target="Bruno",
                message="I think we should cooperate.",
            )
        return Action(agent.name, ActionType.REST)


def test_work_requires_minimum_energy():
    rng = random.Random(1)
    ada = Agent("Ada", money=100, energy=10)
    world = World([ada], WorkPolicy(), rng)

    world.step()

    assert ada.money == 100
    assert any(event.kind == "failed_work" for event in world.events)


def test_talk_preserves_exact_message_in_objective_event():
    rng = random.Random(1)
    world = World([Agent("Ada"), Agent("Bruno")], TalkPolicy(), rng)

    world.step()

    event = next(event for event in world.events if event.kind == "talk")
    assert event.message == "I think we should cooperate."
    assert "I think we should cooperate." in event.summary



def test_cognition_trace_records_both_sides_of_direct_interaction():
    rng = random.Random(1)
    cognition = CognitionEngine()
    world = World(
        [Agent("Ada"), Agent("Bruno")],
        TalkPolicy(),
        rng,
        cognition=cognition,
    )

    world.step()

    talk_traces = [
        trace for trace in cognition.traces
        if trace.event_kind == "talk"
    ]
    assert {trace.observer for trace in talk_traces} == {"Ada", "Bruno"}
    assert all(trace.event_actor == "Ada" for trace in talk_traces)
    assert all(trace.event_target == "Bruno" for trace in talk_traces)
    assert all(
        trace.message == "I think we should cooperate."
        for trace in talk_traces
    )


def test_run_can_write_separate_cognition_jsonl(tmp_path):
    import json
    from src.acl.simulation import run

    event_log = tmp_path / "events.jsonl"
    cognition_log = tmp_path / "cognition.jsonl"

    run(
        turns=1,
        seed=7,
        log_path=event_log,
        cognition_log_path=cognition_log,
    )

    event_rows = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    cognition_rows = [
        json.loads(line)
        for line in cognition_log.read_text(encoding="utf-8").splitlines()
    ]

    assert event_rows
    assert cognition_rows
    assert "observer" not in event_rows[0]
    assert "observer" in cognition_rows[0]
    assert "relationship_delta" in cognition_rows[0]
