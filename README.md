# Artificial Chaos Lab

Small artificial worlds, questionable rules, autonomous agents, and whatever chaos emerges.

The goal is not to prove how humans behave. The goal is to build playful, reproducible multi-agent experiments where the interesting part is **what the agents decide to do once the world starts moving**.

## Core idea

We define:

- a world with explicit rules,
- agents with private goals and limited information,
- a finite set of actions,
- consequences enforced by code,
- an objective event log,
- subjective memories, beliefs and relationships per agent.

The model chooses intentions. **The simulation engine decides what is physically possible and what actually happens.**

An important design rule is that world truth and agent truth are separate: an `Event` records what objectively happened, while each agent may remember and interpret that event differently.

## Experiment 001 — Tiny Town

Five agents live in a tiny economy. They need food and energy, can work, rest, buy food, give money, talk, or try to steal.

The current baseline intentionally uses deterministic/random policies and a heuristic memory interpreter so we can validate the simulation and cognition layers before adding local inference.

Later we will swap those pieces for a local model through Ollama/llama.cpp.

## Architecture

```text
                        ┌──────────── Objective Event Log
                        │
Experiment config       │
      │                 │
      ▼                 │
 Simulation Engine ─────┤
      │                 │
      │                 ▼
      │          Cognition / Perception
      │                 │
      │                 ├── memories
      │                 ├── beliefs
      │                 └── relationships
      │
      └── Policy interface
              │
              ├── RandomPolicy (now)
              └── LocalLLMPolicy (next)
```

## Run

Requires Python 3.11+.

```bash
python -m src.acl.cli --turns 100 --seed 7
```

## Design principles

1. **Rules create possibility, not outcomes.**
2. **Chaos should emerge, not be scripted.**
3. **World truth and agent belief are different things.**
4. **Agents should decide what experiences matter to them.**
5. **Bad actions are allowed when the world permits them; consequences matter more than bans.**
6. **Every experiment must be replayable from objective logs.**
7. **Interesting failures are data.**
8. **This is entertainment/engineering exploration, not social science evidence.**

## Roadmap

- [x] Minimal turn-based simulation engine
- [x] Structured actions and objective event log
- [x] Seeded runs for reproducibility
- [x] Subjective memory, beliefs and relationships
- [x] Memory retrieval/reinforcement baseline
- [ ] Ollama local-model policy
- [ ] LLM-driven subjective interpretation
- [ ] Memory consolidation, forgetting and distortion
- [ ] Spatial 2D map
- [ ] Replay UI with moving agents and interaction lines
- [ ] Metrics dashboard
- [ ] Experiment manifests and batch runs
- [ ] Event/highlight detector for video editing

## Possible future episodes

- One liar among agents that trust everyone
- Two football fan groups that start with no rivalry
- A rumor introduced into an otherwise stable town
- Unequal starting wealth
- Agents allowed to write and vote on laws
- One agent that believes an absurd premise and must navigate contradictory evidence

## License

MIT
