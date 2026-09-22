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

The simulation can run with a deterministic/random baseline or with one local Ollama model shared by all agents. Each agent gets its own private context, memories, beliefs and relationships; the model itself is not duplicated per person.

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
              ├── RandomPolicy
              └── OllamaPolicy
                       │
                       ▼
                 qwen3:14b local
```

## Run

Requires Python 3.11+.

Random baseline:

```bash
python -m src.acl.cli --turns 100 --seed 7
```

Local Qwen through Ollama:

```bash
python -m src.acl.cli --policy ollama --model qwen3:14b --turns 3
```

Start with only a few turns. Five agents across three turns already produce roughly fifteen local-model decisions.

See [docs/LOCAL_OLLAMA.md](docs/LOCAL_OLLAMA.md) for the local setup and information-boundary details.

## Design principles

1. **Rules create possibility, not outcomes.**
2. **Chaos should emerge, not be scripted.**
3. **World truth and agent belief are different things.**
4. **Agents should decide what experiences matter to them.**
5. **Agents are not omniscient; private state stays private.**
6. **Bad actions are allowed when the world permits them; consequences matter more than bans.**
7. **Every experiment must be replayable from objective logs.**
8. **Interesting failures are data.**
9. **This is entertainment/engineering exploration, not social science evidence.**

## Roadmap

- [x] Minimal turn-based simulation engine
- [x] Structured actions and objective event log
- [x] Seeded runs for reproducibility
- [x] Subjective memory, beliefs and relationships
- [x] Memory retrieval/reinforcement baseline
- [x] Ollama local-model policy
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
