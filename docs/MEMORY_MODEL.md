# Subjective memory model

Artificial Chaos Lab keeps two different records on purpose.

## Objective layer — what happened

The world emits immutable `Event` objects.

Example:

```text
turn=42
actor=Bruno
kind=give
target=Pedro
amount=10
```

This is the ground truth used by the experimenter when analysing a run.

## Subjective layer — what an agent thinks happened

Each direct participant perceives the event through a `MemoryInterpreter`.

The interpreter may decide:

- whether the event is worth remembering,
- how the agent describes it,
- who the memory is about,
- how important/emotional it feels,
- whether trust changes,
- whether a belief is formed or reinforced.

That means the same objective event can create different memories in different agents.

## Baseline today

`HeuristicMemoryInterpreter` is intentionally simple and deterministic. It exists so the memory architecture can be tested before local LLM inference is connected.

It should eventually be replaced by something like:

```text
objective event
      ↓
agent identity + current state + prior beliefs
      ↓
local LLM
      ↓
subjective interpretation
      ↓
memory / relationship / belief updates
```

The LLM will not be allowed to rewrite the objective event log.

## Recall

Memories are ranked from a blend of:

- importance,
- emotional intensity,
- recency,
- prior recalls,
- relevance to the current subject.

Recalling a memory updates `last_recalled` and `recall_count`. This lets repeated recollection make some memories easier to surface again later.

Semantic relevance and memory distortion are future milestones.

## Why both layers matter

A future video can compare:

> What actually happened?

against:

> What does Pedro now believe happened?

That gap is one of the main sources of interesting emergent behaviour we want to observe.
